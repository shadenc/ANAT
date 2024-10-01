import cv2
import numpy as np
import json
from speed_detection_module.camera_calibration import CameraCalibration
from speed_detection_module.car_detection import CarDetection
from speed_detection_module.bounding_box_constructor import BoundingBoxConstructor
from speed_detection_module.vehicle_tracker import VehicleTracker
from speed_detection_module.speed_calculator import SpeedCalculator
from speed_detection_module.depth_estimation import DepthEstimationModel
from speed_detection_module.masker import Masker
from speed_detection_module.optical_flow_estimator import OpticalFlowEstimator
from ultralytics import YOLO
from anpr_module.Help_util import assign_car, preprocess_frame, read_license_plate, write_csv
import os
from datetime import datetime
from google.cloud import storage

class IntegratedVideoProcessor:
    def __init__(self, video_path, gcs_bucket_name, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy',
                 detection_confidence=0.4):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.detection_confidence = detection_confidence

        # Google Cloud Storage setup
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket_name)

        # Speed Detection Pipeline components
        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)
        self.flow_estimator = OpticalFlowEstimator()

        # ANPR Components
        self.license_plate_detector = YOLO('models/FT_LPM.pt')  # License plate detection model

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.results = []  # Store results for JSON logging
        self.license_plate_results = {}  # Store license plate detection results

    def save_image_to_gcs(self, image, gcs_path):
        """Saves an image directly to GCS."""
        _, encoded_image = cv2.imencode('.jpg', image)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')
        print(f'Image saved to {gcs_path}')

    def process_video(self):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Capture timestamp
            vis_frame = frame.copy()

            # Vehicle Detection and Speed Calculation
            masked_frame = self.masker.apply_mask(frame)
            depth_map = self.depth_model.estimate_depth(masked_frame)
            flow = self.flow_estimator.estimate_flow(masked_frame)
            vehicle_detections = self.car_detection.detect_cars(masked_frame, self.ipm_matrix, self.calibration, self.detection_confidence)

            # ANPR - License Plate Detection
            license_plate_detections = self.license_plate_detector(frame)[0]
            vehicles_detected = []  # List for vehicle bbox details

            bboxes_3d = []
            for det in vehicle_detections:
                x1, y1, x2, y2, conf, cls = det
                center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
                if np.isnan(center_depth) or np.isinf(center_depth):
                    continue
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)
                    vehicles_detected.append([x1, y1, x2, y2, None, conf])

            # Tracking and Speed Calculation
            tracks = self.tracker.update(bboxes_3d)
            for track_id, track in tracks.items():
                corners_3d = track['bbox_3d']
                speed, confidence = self.speed_calculator.calculate_speed(track_id, np.mean(corners_3d, axis=0),
                                                                          current_time, current_time - 1 / self.fps, unit='km/h')

                # Save car image to GCS
                car_image = frame[int(corners_3d[0][1]):int(corners_3d[1][1]), int(corners_3d[0][0]):int(corners_3d[1][0])]
                car_image_gcs_path = f'car_images/car_{track_id}_{timestamp}.jpg'
                self.save_image_to_gcs(car_image, car_image_gcs_path)

                # Associate License Plates with Vehicles
                for license_plate in license_plate_detections.boxes.data.tolist():
                    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, _ = license_plate
                    x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected)
                    if car_id != -1:
                        # Save license plate image to GCS
                        license_plate_image = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp)]
                        license_plate_image_gcs_path = f'license_plate_images/license_plate_{car_id}_{timestamp}.jpg'
                        self.save_image_to_gcs(license_plate_image, license_plate_image_gcs_path)

                        # OCR for license plate
                        lp_crop = preprocess_frame(frame, x1_lp, y1_lp, x2_lp, y2_lp)
                        license_plate_text = read_license_plate(lp_crop)

                        # Log detection results
                        self.license_plate_results[frame_count] = {
                            'track_id': track_id,
                            'license_plate': {'text': license_plate_text, 'bbox': [x1_lp, y1_lp, x2_lp, y2_lp]},
                            'car_image_gcs_path': car_image_gcs_path,
                            'license_plate_image_gcs_path': license_plate_image_gcs_path,
                            'speed': speed,
                            'timestamp': timestamp
                        }

                # Store results for speed and license plate in JSON format
                self.results.append({
                    'frame': frame_count,
                    'track_id': track_id,
                    'speed': speed if speed is not None else 'N/A',
                    'license_plate': self.license_plate_results.get(frame_count, {}).get('license_plate', 'N/A'),
                    'car_image_gcs_path': car_image_gcs_path,
                    'license_plate_image_gcs_path': license_plate_image_gcs_path,
                    'timestamp': timestamp,
                    'confidence': confidence if confidence is not None else 'N/A'
                })

        # Save results JSON to GCS
        results_json = json.dumps(self.results, indent=2)
        blob = self.bucket.blob('output/speed_estimation_results.json')
        blob.upload_from_string(results_json, content_type='application/json')
        print('Results JSON uploaded to GCS.')

        self.cap.release()
        cv2.destroyAllWindows()

# Running the pipeline
gcs_bucket_name = 'your-gcs-bucket-name'
processor = IntegratedVideoProcessor('input/testing.mp4', gcs_bucket_name)
processor.process_video()
