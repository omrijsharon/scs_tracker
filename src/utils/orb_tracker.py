import numpy as np
import cv2
import warnings


class ORB_tracker:
    def __init__(self, n_features, top_k_features, init_crop_size=101):
        self.orb = cv2.ORB_create(n_features)
        self.top_k_features = top_k_features
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_crop_top_left = None
        self.last_crop_bottom_right = None
        self.last_crop_size = None
        self.next_crop_top_left = None
        self.next_crop_bottom_right = None
        self.next_crop_size = (init_crop_size, init_crop_size)
        self.init_crop_size = init_crop_size
        self.xy = None

    def reset(self, frame, xy):
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.last_crop_top_left = None
        self.last_crop_bottom_right = None
        self.last_crop_size = None
        self.next_crop_top_left = None
        self.next_crop_bottom_right = None
        self.next_crop_size = (self.init_crop_size, self.init_crop_size)
        self.xy = xy
        cropped_frame = self.crop_frame(frame)
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        kp_np = np.array([kp.pt for kp in keypoints])
        #draw circles around keypoints
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(cropped_frame, (int(x), int(y)), 2, (0, 255, 0), 2)
        cv2.imshow('keypoints', cropped_frame)
        self._update_variables(frame, keypoints, descriptors)

    def update(self, frame):
        if self.xy is None:
            raise Exception('Run reset() before update()')
        cropped_frame = self.crop_frame(frame)
        # cropped_frame = frame
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(cropped_frame, (int(x), int(y)), 2, (0, 255, 0), 2)
        cv2.imshow('keypoints', cropped_frame)
        matches = self.matcher.match(self.last_descriptors, descriptors)
        if matches is None: # no matches
            warnings.warn('No matches found')
            self._update_variables(frame, keypoints, descriptors)
            return None
        else: # matches found
            matches = sorted(matches, key=lambda x: x.distance)[:self.top_k_features]
            # prev_pts = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            next_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            self._update_variables(frame, keypoints, descriptors)
            self.xy = (self.last_crop_top_left + next_pts.mean(axis=0).reshape(-1)).astype(int)
            self.next_crop_size = tuple(next_pts.std(axis=0).reshape(-1).astype(int) * 4)
            return (next_pts + self.last_crop_top_left).astype(np.int32)

    @staticmethod
    def descriptors_cosine_similarity(descriptors1, descriptors2):
        return (descriptors1 / np.linalg.norm(descriptors1, axis=1, keepdims=True)) @ (
                    descriptors2 / np.linalg.norm(descriptors2, axis=1, keepdims=True)).T

    def crop_frame(self, frame):
        x, y = self.xy
        self.next_crop_top_left = np.array([x - self.next_crop_size[0] // 2, y - self.next_crop_size[1] // 2])
        self.next_crop_bottom_right = np.array([x + self.next_crop_size[0] // 2 + 1, y + self.next_crop_size[1] // 2 + 1])
        return frame[
               self.next_crop_top_left[1]:self.next_crop_bottom_right[1],
               self.next_crop_top_left[0]:self.next_crop_bottom_right[0]
               ]

    def _update_variables(self, frame, keypoints, descriptors):
        self.last_frame = frame
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        self.last_crop_top_left = self.next_crop_top_left
        self.last_crop_bottom_right = self.next_crop_bottom_right
        self.last_crop_size = self.next_crop_size