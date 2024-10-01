import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import matplotlib.pyplot as plt

class CameraCalibration:
    def __init__(self, model_path='models/vp_using_seg_model_best.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model_path = model_path
        self.model = None
        self.ipm_matrix = None
        self.width = None
        self.height = None
        self.camera_matrix = None
        self.distortion_coeffs = None

    def load_model(self):
        if self.model is None:
            self.model = keras.models.load_model(self.model_path)

    def calibrate_camera(self, frames):
        self.load_model()
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the vanishing points
        vp1 = self.find_vanishing_point(frames[0])
        vp2 = self.find_vanishing_point(frames[-1])
        vp2 = self.orthogonalize_vanishing_points(vp1, vp2)

        self.focal_length = np.sqrt(abs(np.dot(vp1, vp2)))
        vp3 = np.cross(vp1, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [vp1, vp2, vp3]

        # Estimate camera matrix and distortion coefficients
        self.estimate_camera_parameters()

        # Compute IPM matrix
        self.compute_ipm_matrix()

        # Visualize vanishing points
        self.visualize_vanishing_points(frames[0], [vp1, vp2, vp3])

        return self.ipm_matrix

    def orthogonalize_vanishing_points(self, vp1, vp2):
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_vanishing_point(self, frame):
        self.load_model()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        segmentation, vp = self.model.predict(frame)
        vp = vp[0]

        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def compute_ipm_matrix(self):
        src_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        dst_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width * 0.75, 0],
            [self.width * 0.25, 0]
        ])

        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def get_camera_matrix(self):
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated. Call calibrate_camera first.")
        return self.camera_matrix

    def save_calibration(self, filename):
        data = {
            'focal_length': self.focal_length.tolist() if isinstance(self.focal_length, np.ndarray) else self.focal_length,
            'principal_point': self.principal_point,
            'vanishing_points': [vp.tolist() for vp in self.vanishing_points],
            'ipm_matrix': self.ipm_matrix.tolist(),
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'width': self.width,
            'height': self.height
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.focal_length = np.array(data['focal_length'])
        self.principal_point = tuple(data['principal_point'])
        self.vanishing_points = [np.array(vp) for vp in data['vanishing_points']]
        self.ipm_matrix = np.array(data['ipm_matrix'])
        self.camera_matrix = np.array(data['camera_matrix'])
        self.distortion_coeffs = np.array(data['distortion_coeffs'])
        self.width = data['width']
        self.height = data['height']

    def apply_ipm(self, frame):
        if self.ipm_matrix is None:
            raise ValueError("IPM matrix has not been computed. Call calibrate_camera first.")
        return cv2.warpPerspective(frame, self.ipm_matrix, (self.width, self.height))

    def visualize_vanishing_points(self, frame, vanishing_points):
        frame_with_vps = frame.copy()

        for idx, vp in enumerate(vanishing_points):
            x, y = int(vp[0]), int(vp[1])
            cv2.circle(frame_with_vps, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame_with_vps, f'VP{idx+1}', (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for vp in vanishing_points:
            cv2.line(frame_with_vps, (self.width // 4, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (self.width // 2, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (3 * self.width // 4, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(frame_with_vps, cv2.COLOR_BGR2RGB))
        plt.title('Vanishing Points Visualization')
        plt.axis('off')
        plt.show()

    def estimate_camera_parameters(self):
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])
        self.distortion_coeffs = np.zeros(5)

    def undistort_image(self, image):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Camera parameters not calibrated. Call calibrate_camera first.")
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

    def rectify_image(self, image):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Camera parameters not calibrated. Call calibrate_camera first.")

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (self.width, self.height), 1, (self.width, self.height)
        )

        mapx, mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix, (self.width, self.height), 5
        )
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    def calibrate_radiometric(self, frames, exposure_times):
        if len(frames) < 2 or len(frames) != len(exposure_times):
            raise ValueError("At least two frames with corresponding exposure times are needed for radiometric calibration")

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        calibrate_debevec = cv2.createCalibrateDebevec()
        response = calibrate_debevec.process(gray_frames, np.array(exposure_times, dtype=np.float32))

        merge_debevec = cv2.createMergeDebevec()
        radiance = merge_debevec.process(gray_frames, exposure_times, response)

        np.save('camera_response.npy', response)
        np.save('radiance_map.npy', radiance)

        return response, radiance