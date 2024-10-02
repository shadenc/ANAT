import numpy as np
from scipy.optimize import linear_sum_assignment

class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3, max_track_length=30):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.max_track_length = max_track_length
        self.track_id_count = 0

    def update(self, detections_3d):
        self.frame_count += 1

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip or \
               self.frame_count - self.tracks[track_id]['first_seen'] > self.max_track_length:
                del self.tracks[track_id]
            else:
                self.tracks[track_id]['missed_frames'] += 1

        # Match detections to tracks
        if len(detections_3d) > 0 and len(self.tracks) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(detections_3d), len(self.tracks)))
            for i, detection in enumerate(detections_3d):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    iou_matrix[i, j] = self.iou_3d(detection, track['bbox_3d'])

            # Handle NaN and inf values
            iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Match using Hungarian algorithm
            detection_indices, track_indices = linear_sum_assignment(-iou_matrix)

            matched_indices = np.column_stack((detection_indices, track_indices))

            for d, t in matched_indices:
                if iou_matrix[d, t] > 0.3:  # IOU threshold
                    track_id = list(self.tracks.keys())[t]
                    self.tracks[track_id]['bbox_3d'] = detections_3d[d]
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    self.tracks[track_id]['missed_frames'] = 0
                    self.tracks[track_id]['hits'] += 1
                else:
                    self.create_new_track(detections_3d[d])

            # Create new tracks for unmatched detections
            unmatched_detections = set(range(len(detections_3d))) - set(detection_indices)
            for d in unmatched_detections:
                self.create_new_track(detections_3d[d])
        elif len(detections_3d) > 0:
            # If there are no existing tracks, create new tracks for all detections
            for detection in detections_3d:
                self.create_new_track(detection)

        return self.tracks

    def create_new_track(self, detection):
        self.track_id_count += 1
        self.tracks[self.track_id_count] = {
            'bbox_3d': detection,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'missed_frames': 0,
            'hits': 1
        }

    def iou_3d(self, bbox1, bbox2):
        try:
            def volume(bbox):
                return np.prod(np.max(bbox, axis=0) - np.min(bbox, axis=0))

            intersection = np.minimum(np.max(bbox1, axis=0), np.max(bbox2, axis=0)) - np.maximum(np.min(bbox1, axis=0), np.min(bbox2, axis=0))
            intersection = np.maximum(intersection, 0)
            intersection_volume = np.prod(intersection)

            volume1 = volume(bbox1)
            volume2 = volume(bbox2)

            iou = intersection_volume / (volume1 + volume2 - intersection_volume)
            return np.clip(iou, 0, 1)  # Ensure IoU is between 0 and 1
        except Exception as e:
            print(f"Error in IOU calculation: {str(e)}")
            return 0
