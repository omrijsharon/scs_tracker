import numpy as np
import cv2
import warnings
from itertools import compress

from utils.helper_functions import match_ratio_test, filter_unique_matches, kp_mean_and_std


class ORB_tracker:
    def __init__(self, init_crop_size=201):
        self.orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
        self.proximity_radius = 31
        self.std_dist_from_mean = 1
        self.crop_size_scale = 201
        self.min_kp_matches = 10
        self.min_crop_size = 21
        self.max_crop_size = 111
        self.max_des_array_size = 500
        self.xy_smooth_factor = 0.3
        self.last_keypoints = None
        self.last_descriptors = None
        self.descriptors_array = None
        self.descriptors_array_score = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.last_crop_top_left = None
        self.last_crop_bottom_right = None
        self.last_crop_size = None
        self.next_crop_top_left = None
        self.next_crop_bottom_right = None
        self.next_crop_size = (init_crop_size, init_crop_size)
        self.init_crop_size = init_crop_size
        self.xy = None
        self.xy_follow = None
        self.is_initialized = False
        self.is_successful = True

    def reset(self, frame, xy):
        self.last_keypoints = None
        self.last_descriptors = None
        self.last_crop_top_left = None
        self.last_crop_bottom_right = None
        self.last_crop_size = None
        self.descriptors_array = None
        self.descriptors_array_score = None
        self.next_crop_top_left = None
        self.next_crop_bottom_right = None
        self.next_crop_size = (self.init_crop_size, self.init_crop_size)
        self.xy = xy
        self.xy_follow = np.array(xy)
        cropped_frame = self.crop_frame(frame)
        # compute keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        if len(keypoints) > 0:
            # correct the keypoints coordinates using self.next_crop_top_left
            self.correct_keypoints_coordinates(keypoints)
            keypoints, descriptors = self.filter_kp_and_des_by_dist_from_xy(keypoints, descriptors)
            if len(keypoints) > 0:
                keypoints, descriptors = self.filter_kp_and_des_by_std_dist_from_mean(keypoints, descriptors)
                # cv2.imshow('keypoints', cropped_frame)
                self._update_variables(keypoints, descriptors)
                self.is_initialized = True
                self.is_successful = True
            else:
                warnings.warn('No keypoints found')
                self.is_initialized = False
                self.is_successful = False
        else:
            warnings.warn('No keypoints found')
            self.is_initialized = False
            self.is_successful = False

    def update(self, frame):
        if self.xy is None and self.is_initialized:
            raise Exception('Run reset() before update()')
        cropped_frame = self.crop_frame(frame)
        keypoints, descriptors = self.orb.detectAndCompute(cropped_frame, None)
        if len(keypoints) == 0:
            warnings.warn('No keypoints found. Searching in the whole frame')
            self.orb.setMaxFeatures(5000)
            self.orb.setFastThreshold(50)
            keypoints, descriptors = self.orb.detectAndCompute(frame, None)
            # self._update_variables(keypoints, descriptors)
            self.is_successful = False
            # return None, None
        # else:
        #     grid = self.grid_of_keypoints(cropped_frame, keypoints)
        #     self.paste_grid_of_keypoints_on_frame(frame, grid)
        #     keypoints, descriptors = self.filter_keypoints_and_des_by_grid(keypoints, descriptors, grid)
        #     print("len(keypoints): ", len(keypoints))
        # keypoints, descriptors = self.filter_kp_and_des_by_kp_attr(keypoints, descriptors, 'size', 8, 100)
        if self.is_successful:
            keypoints, descriptors = self.filter_kp_and_des_by_std_dist_from_mean(keypoints, descriptors)
        if len(keypoints) == 0:
            warnings.warn('No keypoints found')
            # self._update_variables(keypoints, descriptors)
            # self.is_initialized = False
            self.is_successful = False
            return None, None
        if self.is_successful:
            self.correct_keypoints_coordinates(keypoints)
        if self.descriptors_array is None:
            matches = match_ratio_test(self.matcher, self.last_descriptors, descriptors)
            # matches = self.matcher.match(self.last_descriptors, descriptors)
        else:
            matches = match_ratio_test(self.matcher, self.descriptors_array, descriptors)
            # matches = self.matcher.match(self.descriptors_array, descriptors)
        if len(matches) == 0: # no matches
            warnings.warn('No matches found')
            # self._update_variables(keypoints, descriptors)
            self.is_successful = False
            return None, None
        else: # matches found
            # take only unique matches of trainIdx
            matches = filter_unique_matches(matches)
            # matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 0:
                # filter descriptors by matches
                if self.is_successful:
                    # self.add_to_descriptors_array(descriptors[[m.trainIdx for m in matches]])
                    self.update_descriptors_memory(descriptors[[m.trainIdx for m in matches]])
                # prev_pts = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                mean, std = kp_mean_and_std(keypoints)
                print("num of matches: ", len(matches))
                if len(matches) < self.min_kp_matches:
                    matches = None
                self._update_variables(keypoints, descriptors, matches)
                self.xy = mean.astype(int)
                self.xy_follow = self.xy_smooth_factor * self.xy + (1 - self.xy_smooth_factor) * self.xy_follow
                # self.next_crop_size = tuple(np.clip((self.crop_size_scale * std.reshape(-1)).astype(int), self.min_crop_size, self.max_crop_size))
                self.next_crop_size = (201, 201)
                print("next_crop_size: ",self.next_crop_size)
                self.is_successful = True
                # return next_pts.astype(np.int32), len(matches)
            else:
                warnings.warn('No matches found')
                # self._update_variables(keypoints, descriptors)
                self.is_successful = False
                return None, None

    def grid_of_keypoints(self, cropped_frame, keypoints):
        """
        adds 1 to the pixel of each keypoint in the cropped frame
        :return: grid of keypoints
        """
        grid = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
        for kp in keypoints:
            x, y = kp.pt
            grid[int(y), int(x)] += 1
        # renormalize grid to be between 0 and 255 as uint8
        for i in range(5):
            grid = (grid / grid.max() * 255).astype(np.uint8)
            grid = cv2.GaussianBlur(grid, (21, 21), 0)
        grid = (grid / grid.max() * 255).astype(np.uint8)
        # use a threshold to of 213 to grid
        grid[grid < 205] = 0
        # find countours of grid
        contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        countours_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        # sort contours by area
        contours = [cnt for _, cnt in sorted(zip(countours_areas, contours), key=lambda pair: pair[0])]
        # find the 2 contours with the largest area and fill its area with 255
        grid = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
        if len(countours_areas) > 0:
            grid = cv2.drawContours(grid, contours[-3:], -1, 255, -1)
        return grid

    def paste_grid_of_keypoints_on_frame(self, frame, grid):
        """
        taking the grid_of_keypoints and pastes it on the real position in the frame using next_crop_top_left
        """
        frame[self.next_crop_top_left[1]:self.next_crop_bottom_right[1], self.next_crop_top_left[0]:self.next_crop_bottom_right[0]] *= 0
        frame[self.next_crop_top_left[1]:self.next_crop_bottom_right[1], self.next_crop_top_left[0]:self.next_crop_bottom_right[0]] += grid

    def filter_keypoints_and_des_by_grid(self, keypoints, descriptors, grid):
        """
        filters the keypoints and descriptors by the grid
        """
        mask = np.zeros(len(keypoints), dtype=bool)
        for kp in keypoints:
            x, y = kp.pt
            if grid[int(y), int(x)] > 0:
                mask[keypoints.index(kp)] = True
        return list(compress(keypoints, mask)), descriptors[mask]

    def crop_frame(self, frame):
        x, y = self.xy
        self.next_crop_top_left = np.array([x - self.next_crop_size[0] // 2, y - self.next_crop_size[1] // 2])
        self.next_crop_bottom_right = np.array([x + self.next_crop_size[0] // 2 + 1, y + self.next_crop_size[1] // 2 + 1])
        return frame[
               self.next_crop_top_left[1]:self.next_crop_bottom_right[1],
               self.next_crop_top_left[0]:self.next_crop_bottom_right[0]
               ]

    def calc_dist_of_kp_from_xy(self, keypoints):
        return np.linalg.norm(np.array([kp.pt for kp in keypoints]) - self.xy, axis=1)

    def filter_kp_and_des_by_dist_from_xy(self, keypoints, descriptors):
        dist = self.calc_dist_of_kp_from_xy(keypoints)
        return list(compress(keypoints, dist < self.proximity_radius)), descriptors[dist < self.proximity_radius]

    def filter_kp_and_des_by_std_dist_from_mean(self, keypoints, des):
        mean, std = kp_mean_and_std(keypoints)
        dist = np.linalg.norm(np.array([kp.pt for kp in keypoints]) - mean, axis=1)
        # mask = np.any((dist.reshape(-1, 1) < self.std_dist_from_mean * std), axis=1)
        mask = np.any((dist.reshape(-1, 1) < self.proximity_radius), axis=1)
        return list(compress(keypoints, mask)), des[mask]

    def filter_kp_and_des_by_kp_attr(self, keypoints, des, attr, min_val, max_val):
        # attr of kp can be 'response', 'octave', 'class_id', 'angle', 'size'
        mask = np.any((np.array([getattr(kp, attr) for kp in keypoints]).reshape(-1, 1) > min_val) & (np.array([getattr(kp, attr) for kp in keypoints]).reshape(-1, 1) < max_val), axis=1)
        return list(compress(keypoints, mask)), des[mask]

    def draw_crop_rect_on_frame(self, frame):
        cv2.rectangle(frame, tuple(self.next_crop_top_left), tuple(self.next_crop_bottom_right), (0, 255, 0), 2)

    def draw_keypoints_on_frame(self, frame): # using cv2.drawKeypoints
        cv2.drawKeypoints(frame, self.last_keypoints, frame, color=(0, 255, 0), flags=4)

    def draw_xy_on_frame(self, frame):
        cv2.circle(frame, tuple(self.xy), 2, (0, 255, 0), 2)

    def draw_all_on_frame(self, frame):
        if self.is_successful:
            self.draw_crop_rect_on_frame(frame)
            self.draw_keypoints_on_frame(frame)
            self.draw_xy_on_frame(frame)
            self.draw_cross_on_xy_follow(frame, 5, 10, 5, 2)


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

    def add_to_descriptors_array(self, matches_descriptors):
        """
        check whether the descriptors are already in the descriptors_array, if not, concatinates them to the descriptors_array.
        the self.descriptors_array_score counts how many times each descriptor was matched.
        """
        if self.descriptors_array is None:
            self.descriptors_array = matches_descriptors
            self.descriptors_array_score = np.ones(len(matches_descriptors))
        else:
            # check if the descriptors are already in the descriptors_array.
            # if it does, add 1 to the score of the descriptor in descriptors_array_score.
            # if it doesn't, concatinates them to the descriptors_array and add 1 to the score of the descriptor in descriptors_array_score.
            for des in matches_descriptors:
                idx = np.argwhere(self.descriptors_array == des)[:-1, 1]
                if len(idx) > 0:
                    self.descriptors_array_score[idx] += 1
                else:
                    self.descriptors_array = np.concatenate((self.descriptors_array, des.reshape(1, -1)), axis=0)
                    self.descriptors_array_score = np.concatenate((self.descriptors_array_score, np.array([1])), axis=0)

            if len(self.descriptors_array) > self.max_des_array_size:
                # sort descriptors_array_score and descriptors_array by descriptors_array_score
                self.descriptors_array_score, self.descriptors_array = zip(*sorted(zip(self.descriptors_array_score, self.descriptors_array), key=lambda pair: pair[0]))
                # remove the first len(self.descriptors_array) - self.max_des_array_size descriptors
                self.descriptors_array = np.array(self.descriptors_array[-self.max_des_array_size:])
                self.descriptors_array_score = np.array(self.descriptors_array_score[-self.max_des_array_size:])
            print("max score: ", self.descriptors_array_score.max())
        return self.descriptors_array, self.descriptors_array_score

    def draw_cross_on_xy_follow(self, img, radius, outer_radius, cross_size, thickness, color_black=(0, 0, 0), color_white=(255, 255, 255)):
        # draw a cross at the middle of the screen but without its center. make it out of 4 lines
        y = int(self.xy_follow[0])
        x = int(self.xy_follow[1])
        cv2.line(img, (y - cross_size, x), (y - radius - outer_radius, x),
                color_black, thickness + 2)
        cv2.line(img, (y + cross_size, x), (y + radius + outer_radius, x),
                color_black, thickness + 2)
        cv2.line(img, (y, x - cross_size), (y, x - radius - outer_radius),
                color_black, thickness + 2)
        cv2.line(img, (y, x + cross_size), (y, x + radius + outer_radius),
                color_black, thickness + 2)
        cv2.line(img, (y - cross_size, x), (y - radius - outer_radius, x),
                color_white, thickness)
        cv2.line(img, (y + cross_size, x), (y + radius + outer_radius, x),
                color_white, thickness)
        cv2.line(img, (y, x - cross_size), (y, x - radius - outer_radius),
                color_white, thickness)
        cv2.line(img, (y, x + cross_size), (y, x + radius + outer_radius),
                color_white, thickness)
        cv2.circle(img, (y, x), radius, color_black, thickness + 1)
        cv2.circle(img, (y, x), radius - 1, color_white, thickness)

    def update_descriptors_memory(self, matched_descriptors):
        if self.descriptors_array is None:
            self.descriptors_array = matched_descriptors
            self.descriptors_array_score = np.ones(len(matched_descriptors))
        else:
            for descriptor in matched_descriptors:
                # Convert descriptor to a hashable type
                descriptor_tuple = tuple(descriptor)

                # Check if the descriptor already exists in the buffer
                found = False
                for i, existing_descriptor in enumerate(self.descriptors_array):
                    if tuple(existing_descriptor) == descriptor_tuple:
                        # Descriptor found, increment score
                        self.descriptors_array_score[i] += 1
                        found = True
                        break

                if not found:
                    # Descriptor not found, add to buffer and initialize score
                    self.descriptors_array = np.append(self.descriptors_array, [descriptor], axis=0)
                    self.descriptors_array_score = np.append(self.descriptors_array_score, 1)

            # Check if buffer size exceeds the maximum allowed
            if len(self.descriptors_array) > self.max_des_array_size:
                # Get indices of descriptors sorted by score
                sorted_indices = np.argsort(self.descriptors_array_score)

                # Keep only the top descriptors based on max_des_array_size
                self.descriptors_array = self.descriptors_array[sorted_indices[-self.max_des_array_size:]]
                self.descriptors_array_score = self.descriptors_array_score[sorted_indices[-self.max_des_array_size:]]