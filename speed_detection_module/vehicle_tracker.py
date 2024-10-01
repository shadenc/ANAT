import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

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
                self.tracks[track_id]['kf'].predict()

        # Match detections to tracks
        if len(detections_3d) > 0 and len(self.tracks) > 0:
            # Compute distance matrix
            distance_matrix = np.zeros((len(detections_3d), len(self.tracks)))
            for i, detection in enumerate(detections_3d):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    distance_matrix[i, j] = self.mahalanobis_distance(detection[:3], track['kf'])

            # Handle NaN and inf values
            distance_matrix = np.nan_to_num(distance_matrix, nan=np.inf, posinf=np.inf, neginf=np.inf)

            # Match using Hungarian algorithm
            detection_indices, track_indices = linear_sum_assignment(distance_matrix)

            matched_indices = np.column_stack((detection_indices, track_indices))

            for d, t in matched_indices:
                if distance_matrix[d, t] < 5.0:  # Mahalanobis distance threshold
                    track_id = list(self.tracks.keys())[t]
                    self.tracks[track_id]['kf'].update(detections_3d[d][:3])
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
        kf = self.init_kalman_filter(detection[:3])
        self.tracks[self.track_id_count] = {
            'kf': kf,
            'bbox_3d': detection,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'missed_frames': 0,
            'hits': 1
        }

    def init_kalman_filter(self, initial_state):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([*initial_state, 0, 0, 0])  # Initial state (x, y, z, vx, vy, vz)
        kf.F = np.array([  # State transition matrix
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([  # Measurement function
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        kf.R = np.eye(3) * 0.1  # Measurement noise
        kf.P *= 1000  # Covariance matrix
        kf.Q = np.eye(6) * 0.1  # Process noise
        return kf

    def mahalanobis_distance(self, detection, kf):
        y = detection - kf.x[:3]
        S = kf.H @ kf.P @ kf.H.T + kf.R
        return np.sqrt(y.T @ np.linalg.inv(S) @ y)

    def get_speed(self, track_id):
        if track_id in self.tracks:
            kf = self.tracks[track_id]['kf']
            vx, vy, vz = kf.x[3:6]
            return np.sqrt(vx**2 + vy**2 + vz**2)
        return None

    def get_track_info(self, track_id):
        if track_id in self.tracks:
            track = self.tracks[track_id]
            return {
                'position': track['kf'].x[:3],
                'velocity': track['kf'].x[3:6],
                'bbox_3d': track['bbox_3d'],
                'last_seen': track['last_seen'],
                'hits': track['hits']
            }
        return None