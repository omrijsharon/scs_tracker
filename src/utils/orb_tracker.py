import numpy as np
import cv2
import warnings


class ORB_tracker:
    def __init__(self, init_crop_size=101):
        self.orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
        self.crop_size_scale = 16
        self.min_kp_matches = 15
        self.min_crop_size = 31
        self.max_crop_size = 131
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
        self.is_initialized = False

    def reset(self, frame, xy):
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
        # compute keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        # correct the keypoints coordinates using self.next_crop_top_left
        self.correct_keypoints_coordinates(keypoints)
        # cv2.imshow('keypoints', cropped_frame)
        self._update_variables(keypoints, descriptors)
        self.is_initialized = True

    def update(self, frame):
        if self.xy is None and self.is_initialized:
            raise Exception('Run reset() before update()')
        cropped_frame = self.crop_frame(frame)
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        print("len(keypoints): ", len(keypoints))
        if len(keypoints) == 0:
            warnings.warn('No keypoints found')
            # self._update_variables(keypoints, descriptors)
            self.is_initialized = False
            return None, None
        self.correct_keypoints_coordinates(keypoints)
        matches = self.matcher.match(self.last_descriptors, descriptors)
        if matches is None: # no matches
            warnings.warn('No matches found')
            # self._update_variables(keypoints, descriptors)
            self.is_initialized = False
            return None, None
        else: # matches found
            matches = sorted(matches, key=lambda x: x.distance)# [:self.top_k_features]
            # prev_pts = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            next_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if len(matches) < self.min_kp_matches:
                matches = None
            self._update_variables(keypoints, descriptors, matches)
            self.xy = (next_pts.mean(axis=0).reshape(-1)).astype(int)
            self.next_crop_size = tuple(np.clip((self.crop_size_scale * next_pts.std(axis=0).reshape(-1) * 10).astype(int), self.min_crop_size, self.max_crop_size))
            print("next_crop_size: ",self.next_crop_size)
            # return next_pts.astype(np.int32), len(matches)

    def update_cosim(self, frame):
        print("!")
        if self.xy is None:
            raise Exception('Run reset() before update()')
        cropped_frame = self.crop_frame(frame)
        # cropped_frame = frame
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        print(len(descriptors))
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(cropped_frame, (int(x), int(y)), 2, (0, 255, 0), 2)
        cv2.imshow('keypoints', cropped_frame)
        matches, cos_dist = self.match_descriptors(self.last_descriptors, descriptors)
        # matches = sorted(matches, key=lambda x: x.distance)[:self.top_k_features]
        # prev_pts = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        next_pts = np.float32([keypoints[m].pt for m in matches]).reshape(-1, 1, 2)
        self._update_variables(frame, keypoints, descriptors)
        self.xy = (self.last_crop_top_left + next_pts.mean(axis=0).reshape(-1)).astype(int)
        # self.next_crop_size = tuple(next_pts.std(axis=0).reshape(-1).astype(int) * 10)
        print(self.next_crop_size)
        return (next_pts + self.last_crop_top_left).astype(np.int32)

    @staticmethod
    def descriptors_cosine_similarity(descriptors1, descriptors2):
        return (descriptors1 / np.linalg.norm(descriptors1, axis=1, keepdims=True)) @ (
                    descriptors2 / np.linalg.norm(descriptors2, axis=1, keepdims=True)).T

    def match_descriptors(self, descriptors1, descriptors2):
        similarity_matrix = self.descriptors_cosine_similarity(descriptors1, descriptors2)
        return np.argmax(similarity_matrix, axis=1), np.max(similarity_matrix, axis=1)

    def crop_frame(self, frame):
        x, y = self.xy
        self.next_crop_top_left = np.array([x - self.next_crop_size[0] // 2, y - self.next_crop_size[1] // 2])
        self.next_crop_bottom_right = np.array([x + self.next_crop_size[0] // 2 + 1, y + self.next_crop_size[1] // 2 + 1])
        return frame[
               self.next_crop_top_left[1]:self.next_crop_bottom_right[1],
               self.next_crop_top_left[0]:self.next_crop_bottom_right[0]
               ]

    def draw_crop_rect_on_frame(self, frame):
        x, y = self.xy
        cv2.rectangle(frame, tuple(self.next_crop_top_left), tuple(self.next_crop_bottom_right), (0, 255, 0), 2)

    def draw_keypoints_on_frame(self, frame): # using cv2.drawKeypoints
        cv2.drawKeypoints(frame, self.last_keypoints, frame, color=(0, 255, 0), flags=4)

    def draw_xy_on_frame(self, frame):
        cv2.circle(frame, tuple(self.xy), 2, (0, 255, 0), 2)

    def draw_all_on_frame(self, frame):
        self.draw_crop_rect_on_frame(frame)
        self.draw_keypoints_on_frame(frame)
        self.draw_xy_on_frame(frame)


    def correct_keypoints_coordinates(self, keypoints):
        for kp in keypoints:
             kp.pt = tuple(np.array(kp.pt) + self.next_crop_top_left)
        return keypoints

    def set_orb_params(self, **orb_params):
        if 'maxFeatures' in orb_params:
            self.orb.setMaxFeatures(orb_params['maxFeatures'])
        if 'scaleFactor' in orb_params:
            self.orb.setScaleFactor(orb_params['scaleFactor'])
        if 'nLevels' in orb_params:
            self.orb.setNLevels(orb_params['nLevels'])
        if 'edgeThreshold' in orb_params:
            self.orb.setEdgeThreshold(orb_params['edgeThreshold'])
        if 'firstLevel' in orb_params:
            self.orb.setFirstLevel(orb_params['firstLevel'])
        if 'WTA_K' in orb_params:
            self.orb.setWTA_K(orb_params['WTA_K'])
        if 'patchSize' in orb_params:
            self.orb.setPatchSize(orb_params['patchSize'])
        if 'fastThreshold' in orb_params:
            self.orb.setFastThreshold(orb_params['fastThreshold'])

    def _update_variables(self, keypoints, descriptors, matches=None):
        # self.last_keypoints = keypoints
        # self.last_descriptors = descriptors
        if matches is None:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
        else: # get only the keypoints and descriptors that were matched
            self.last_keypoints = tuple([keypoints[m.trainIdx] for m in matches])
            self.last_descriptors = np.array([descriptors[m.trainIdx] for m in matches])
        self.last_crop_top_left = self.next_crop_top_left
        self.last_crop_bottom_right = self.next_crop_bottom_right
        self.last_crop_size = self.next_crop_size