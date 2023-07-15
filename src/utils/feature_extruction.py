import cv2
import numpy as np
from functools import partial

class FeatureDetector:
    def __init__(self, frame_shape, method, detector_settings_center, detector_settings_surrounding, qualityLevel=0.01, minDistance=10):
        self.height, self.width = frame_shape
        self.half_height, self.half_width = self.height // 2, self.width // 2
        self.quarter_height, self.quarter_width = self.height // 4, self.width // 4
        self.method = method
        self.detector_settings_center = detector_settings_center
        self.detector_settings_surrounding = detector_settings_surrounding
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance

        if method.lower() == "shi-tomasi":
            # use partial to create a function with the detector_settings as default arguments
            self.detectors = [partial(cv2.goodFeaturesToTrack, **detector_settings_center) if i==4 else partial(cv2.goodFeaturesToTrack, **detector_settings_surrounding) for i in range(9)]
        elif method.lower() == "fast":
            self.detectors = [cv2.FastFeatureDetector_create(**detector_settings_center) if i==4 else cv2.FastFeatureDetector_create(**detector_settings_surrounding) for i in range(9)]
        elif method.lower() == "orb":
            self.detectors = [cv2.ORB_create(**detector_settings_center) if i==4 else cv2.ORB_create(**detector_settings_surrounding) for i in range(9)]
        else:
            raise Exception("Unknown method")

    def detect(self, frame, detector):
        if self.method.lower() in ["shi-tomasi"]:
            # For Shi-Tomasi, we use the goodFeaturesToTrack function
            corners = detector(frame)
            if corners is not None:
                corners = corners.reshape(-1, 2).astype(np.int0)
            return corners
        else:
            # For FAST, ORB, we find the keypoints and convert them to the required format
            keypoints = detector.detect(frame, None)
            return np.int0([kp.pt for kp in keypoints])

    def divide_frame(self, frame):
        segments = []
        coordinates = []

        sections = [
            (0, self.quarter_height, 0, self.quarter_width),
            (0, self.quarter_height, self.quarter_width, self.quarter_width + self.half_width),
            (0, self.quarter_height, self.quarter_width + self.half_width, self.width),
            (self.quarter_height, self.quarter_height + self.half_height, 0, self.quarter_width),
            (self.quarter_height, self.quarter_height + self.half_height, self.quarter_width, self.quarter_width + self.half_width),
            (self.quarter_height, self.quarter_height + self.half_height, self.quarter_width + self.half_width, self.width),
            (self.quarter_height + self.half_height, self.height, 0, self.quarter_width),
            (self.quarter_height + self.half_height, self.height, self.quarter_width, self.quarter_width + self.half_width),
            (self.quarter_height + self.half_height, self.height, self.quarter_width + self.half_width, self.width)
        ]

        for y1, y2, x1, x2 in sections:
            segment = frame[y1:y2, x1:x2]
            segments.append(segment)
            coordinates.append((y1, x1))

        return segments, coordinates

    def detect_in_segments(self, frame):
        segments, coordinates = self.divide_frame(frame)
        all_corners = []

        for detector, segment, coordinate in zip(self.detectors, segments, coordinates):
            corners = self.detect(segment, detector)
            if corners is not None:
                # Offset corners with coordinates to map back onto original frame
                corners += np.array(coordinate)
                all_corners.append(corners)

        return np.vstack(all_corners) if len(all_corners) > 0 else None

if __name__ == '__main__':
    # Test
    image = cv2.imread('test.jpg', 0)
    for method in ["shi-tomasi", "fast", "orb"]:
        detector = FeatureDetector(method)
        features = detector.detect(image)
        print(f"Found {len(features)} features using {method}")
