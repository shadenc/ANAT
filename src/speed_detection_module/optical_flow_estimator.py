import cv2
import numpy as np

class OpticalFlowEstimator:
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def estimate_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            return None

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if self.prev_points is None:
                return None

        next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        flow = good_new - good_old

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)

        return flow

    def visualize_flow(self, frame, flow):
        if flow is None or len(flow) == 0:
            return frame

        mask = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(self.prev_points, self.prev_points - flow)):
            a, b = new[0]  # Access x, y directly
            c, d = old[0]
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        return img

    def get_flow_in_roi(self, flow, roi):
        if flow is None or len(flow) == 0:
            return np.array([])
        x, y, w, h = roi
        mask = np.zeros(flow.shape[0], dtype=bool)
        for i, point in enumerate(self.prev_points):
            px, py = point.ravel()
            if x <= px < x + w and y <= py < y + h:
                mask[i] = True
        return flow[mask]