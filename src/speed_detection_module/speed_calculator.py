import numpy as np
from scipy.signal import savgol_filter

class SpeedCalculator:
    def __init__(self, smoothing_window=5, speed_confidence_threshold=0.8, max_history=100):
        self.previous_positions = {}
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.speed_confidence_threshold = speed_confidence_threshold
        self.max_history = max_history

    def calculate_speed(self, track_id, current_position, current_time, previous_time, unit='m/s'):
        try:
            if track_id in self.previous_positions:
                previous_position = self.previous_positions[track_id]
                time_diff = current_time - previous_time

                if time_diff > 0:
                    displacement = np.linalg.norm(current_position - previous_position)
                    speed = displacement / time_diff

                    # Apply a simple low-pass filter
                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []

                    if len(self.speed_history[track_id]) > 0:
                        speed = 0.7 * speed + 0.3 * self.speed_history[track_id][-1]

                    self.speed_history[track_id].append(speed)

                    # Limit history size
                    if len(self.speed_history[track_id]) > self.max_history:
                        self.speed_history[track_id] = self.speed_history[track_id][-self.max_history:]

                    if len(self.speed_history[track_id]) >= self.smoothing_window:
                        smoothed_speed = np.median(self.speed_history[track_id][-self.smoothing_window:])
                        speed_confidence = self.calculate_speed_confidence(self.speed_history[track_id])
                    else:
                        smoothed_speed = speed
                        speed_confidence = 0.5

                    self.previous_positions[track_id] = current_position
                    return self.convert_speed(smoothed_speed, unit), speed_confidence
            else:
                self.previous_positions[track_id] = current_position

            return None, 0.0
        except Exception as e:
            print(f"Error in speed calculation: {str(e)}")
            return None, 0.0

    def calculate_speed_confidence(self, speed_history):
        if len(speed_history) < 2:
            return 0.5

        cv = np.std(speed_history) / np.mean(speed_history)
        confidence = 1 / (1 + cv)
        return min(confidence, self.speed_confidence_threshold)

    def convert_speed(self, speed, unit):
        if unit == 'm/s':
            return speed
        elif unit == 'km/h':
            return speed * 3.6
        elif unit == 'mph':
            return speed * 2.237
        else:
            raise ValueError(f"Unsupported unit: {unit}")