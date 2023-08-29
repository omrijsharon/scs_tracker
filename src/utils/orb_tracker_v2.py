import cv2 as cv
import numpy as np
from utils.descriptor_buffer_manager import AdaptiveDescriptorBuffer
from utils.helper_functions import crop_frame, correct_kp_coordinates, filter_kp_by_std_dist_from_mean


class ORB_trackerV2:
    def __init__(self, n_closest_kp):
        self.n_closest_kp = n_closest_kp
        self.crop_scale = 70
        self.orb = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        self.descriptor_buffer = AdaptiveDescriptorBuffer(matcher, n_closest_kp, n_closest_kp//2, ratio_threshold=0.85)
        self.is_initialized = False
        self.is_successful = False
        self.xy = None
        self.xy_size = None
        self.top_left = None
        self.crop_size = None
        self.kp = None

    def reset(self, frame, xy): # gets the entire frame and not only a crop
        self.descriptor_buffer.clear()
        self.orb.setMaxFeatures(10000)
        self.orb.setFastThreshold(12)
        self.kp, des = self.orb.detectAndCompute(frame, None)
        if len(self.kp) > 0:
            self.descriptor_buffer.reset(xy, self.kp, des)
            self.is_initialized = True
            self.is_successful = True
            self.xy, self.xy_size = self.descriptor_buffer.mean, self.descriptor_buffer.std
            self.crop_size = np.array(self.xy_size * self.crop_scale).astype(np.int)

    def update(self, frame):
        assert self.is_initialized, "ORB_trackerV2 is not initialized. call reset() first."
        if self.is_successful:
            print("crop size:", self.crop_size)
            self.crop_size = np.clip(self.crop_size, a_min=11, a_max=101)
            cropped_frame, self.top_left = crop_frame(frame, self.xy, self.crop_size)
            self.kp, des = self.orb.detectAndCompute(cropped_frame, None)
            self.kp = correct_kp_coordinates(self.kp, self.top_left)
        else:
            self.orb.setMaxFeatures(10000)
            self.orb.setFastThreshold(12)
            self.kp, des = self.orb.detectAndCompute(frame, None)
        if len(self.kp) > 0:
            self.is_successful = self.descriptor_buffer.update(self.kp, des)
            if self.is_successful:
                self.xy, self.xy_size = self.descriptor_buffer.mean, self.descriptor_buffer.std
                self.kp = self.descriptor_buffer.kp
                self.kp = filter_kp_by_std_dist_from_mean(self.kp, self.xy, self.xy_size, n_std=0.7)
                if np.any(self.xy_size == 0):
                    self.xy_size = np.array([51, 51])
                self.crop_size = np.array(self.xy_size * self.crop_scale).astype(np.int)
                self.crop_size = np.clip(self.crop_size, a_min=11, a_max=101)
                self.is_successful = True
        else:
            self.is_successful = False

    def color_is_successful(self):
        if self.is_successful:
            return (0, 255, 0)
        return (0, 0, 255)

    def draw_rect_around_cropped_frame(self, frame):
        assert self.is_initialized, "ORB_trackerV2 is not initialized. call reset() first."
        cv.rectangle(frame, tuple(self.top_left), tuple(self.top_left + self.crop_size), self.color_is_successful(), 2)

    def draw_cross_on_xy(self, frame):
        assert self.is_initialized, "ORB_trackerV2 is not initialized. call reset() first."
        cv.drawMarker(frame, tuple(self.xy.astype(np.int)), self.color_is_successful(), cv.MARKER_CROSS, 20, 2)

    def draw_kp(self, frame):
        assert self.is_initialized, "ORB_trackerV2 is not initialized. call reset() first."
        cv.drawKeypoints(frame, self.kp, frame, flags=0)

    def draw_all_on_frame(self, frame):
        assert self.is_initialized, "ORB_trackerV2 is not initialized. call reset() first."
        self.draw_kp(frame)
        self.draw_rect_around_cropped_frame(frame)
        self.draw_cross_on_xy(frame)
