import cv2
import numpy as np
from collections import Counter

from utils.helper_functions import crop_frame, softmax, calc_scs, local_sum, get_crop_actual_shape


# Improvments:
# 1. make a kernel for each color and take the result with the highest log_max_change
# 2. make kernels in many sizes and take the result with the highest log_max_change
# 3. add orb keypoints to the scs filter for better target aquasition and for better tracking
# 4. check weather the scs filter is better than the orb keypoints or not: hint, i already checked it and the scs filter is better.
# 5. adding a gaussian blur to any of the frame manipulation make the tracker worse. so do not use it at all.
# 6. try to add octaves to the scs filter like in SIFT. to implement this you need to add a gaussian blur to the frame and then subtract the blurred frame from the original frame.


class SCS_Quadrangle_Tracker:
    def __init__(self, kernel_size, crop_size, square_size, nn_size=3, max_diffusion_radius=11, p=3, q=1e-9, temperature=0.05, max_velocity=50):
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert kernel_size < crop_size, "kernel_size must be smaller than crop_size"
        assert square_size < crop_size, "square_size must be smaller than crop_size"
        self.crop_size = int(crop_size)
        self.half_square_size = int(np.round(square_size/2))
        # self.min_crop_size = int(2 * self.kernel_size)
        self.min_crop_size = int(self.crop_size//2)
        self.max_diffusion_radius = int(max_diffusion_radius)
        self.nn_size = int(nn_size)
        self.log_max_change_threshold = -12
        self.kernel = None
        self.frame_size = None
        self.last_coordinates = None
        self.coordinates = None
        self.kernel_coordinates = None
        self.velocity = None
        self.top_left = None
        self.bottom_right = None
        self.filtered_scs_frame = None
        self.filtered_scs_softmax_frame = None
        self.max_change = None
        self.max_velocity = max_velocity  # pixels per frame
        self.p = p
        self.q = q
        self.temperature = temperature
        self.is_initialized = False
        self.is_successful = False
        self.color = (0, 255, 0)

    def change_kernel_size(self, kernel_size, frame):
        self.kernel_size = kernel_size
        self.create_kernel(frame, self.coordinates)

    def set_crop_size(self,crop_size):
        self.crop_size = crop_size

    def set_nn_size(self,nn_size):
        self.nn_size = nn_size

    def set_p(self,p):
        self.p = p

    def set_q(self,q):
        self.q = q

    def set_temperature(self,temperature):
        self.temperature = temperature

    def set_max_velocity(self,max_velocity):
        self.max_velocity = max_velocity

    def set_max_diffusion_radius(self,max_diffusion_radius):
        self.max_diffusion_radius = max_diffusion_radius

    def set_hue(self, hue):
        # sets self.color to hue, with maximum saturation and value
        hue %= 180
        self.color = tuple(cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0])

    def set_frame_size(self, frame):
        self.frame_size = frame.shape

    def reset_kernel_coordinates(self, yx):
        # reset the kernel coordinates to be yx[0] - self.half_square_size, yx[0] + self.half_square_size, yx[1] - self.half_square_size, yx[1] + self.half_square_size
        self.kernel_coordinates = np.zeros((4, 2), dtype=np.int32)
        self.kernel_coordinates[0] = yx - self.half_square_size # this is the top left corner
        self.kernel_coordinates[1] = [yx[0] - self.half_square_size, yx[1] + self.half_square_size] # this is the bottom left corner
        self.kernel_coordinates[2] = yx + self.half_square_size # this is the bottom right corner
        self.kernel_coordinates[3] = [yx[0] + self.half_square_size, yx[1] - self.half_square_size] # this is the top right corner

    def reset(self, frame, xy):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_size = gray.shape
        self.velocity = np.zeros(2)
        self.coordinates = np.array(xy[::-1])
        self.reset_kernel_coordinates(self.coordinates)
        self.create_kernel(gray, self.kernel_coordinates)
        self.last_coordinates = np.array(self.coordinates)
        self.is_initialized = True
        self.is_successful = True

    def create_kernel(self, frame, kernel_coordinates):
        # make sure frame is grayscale
        # create multiple kernels in all the sized in self.kernel_size
        kernels = []
        for kernel_coord in kernel_coordinates:
            kernel, _, _ = crop_frame(frame, kernel_coord, self.kernel_size)
            kernel /= (np.linalg.norm(kernel) + 1e-9)
            kernels.append(kernel)
        self.kernel = np.array(kernels, dtype=object)

    def update(self, frame):
        assert self.kernel is not None, "kernel is None. call reset() first."
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_frame, self.top_left, self.bottom_right = crop_frame(gray, self.coordinates, self.crop_size)
        # check if cropped_frame shape is larger than the minimum crop size
        if cropped_frame.shape[0] < self.min_crop_size or cropped_frame.shape[1] < self.min_crop_size:
            print("SCS_tracker: crop is too small")
            self.is_successful = False
            return False
        # calculate the SCS filter
        try:
            # gaussian blur to cropped_frame for better results.
            # cropped_frame = cv2.GaussianBlur(cropped_frame, (self.max_diffusion_radius, self.max_diffusion_radius), 0)
            # cropped_frame = cropped_frame - blurred_cropped_frame
            max_idx = np.zeros((len(self.kernel), 2), dtype=np.int32)
            max_change = np.zeros(len(self.kernel))
            for i, kernel in enumerate(self.kernel):
                filtered_scs_frame = calc_scs(cropped_frame, kernel.astype(np.float32), p=self.p, q=self.q)
                max_idx[i], max_change[i] = self.find_max(filtered_scs_frame)
                self.kernel_coordinates[i] = max_idx[i] + self.top_left
                max_change[i] = np.log(max_change[i])
            self.max_change = max_change.max()
        except:
            print("SCS_tracker: calc_scs failed")
            print("crop shape:", cropped_frame.shape)
            self.is_successful = False
            return False

        print("log self.max_change:", self.max_change)
        if self.max_change < self.log_max_change_threshold:
            print("SCS_tracker: max_change is too low")
            self.is_successful = False
            return False
        self.last_coordinates = self.coordinates
        # find the mean of kernel_coordinates
        self.coordinates = self.kernel_coordinates.mean(axis=0)
        # calculate the velocity
        self.velocity = self.coordinates - self.last_coordinates
        # clip the velocity
        self.velocity = np.clip(self.velocity, a_min=-self.max_velocity, a_max=self.max_velocity)
        # create the kernel
        self.create_kernel(gray, self.kernel_coordinates)
        expected_coordinates = self.coordinates + self.velocity
        if not self.is_in_frame(expected_coordinates):
            print("SCS_tracker: expected_coordinates is out of frame: ", expected_coordinates, "  frame size: ", self.frame_size)
            self.is_successful = False
            return False
        return True

    def find_max(self, filtered_scs_frame):
        self.filtered_scs_softmax_frame = softmax(filtered_scs_frame / self.temperature)
        # apply gaussian blur to filtered_scs_softmax_frame
        # self.filtered_scs_softmax_frame = cv2.GaussianBlur(self.filtered_scs_softmax_frame, (self.nn_size, self.nn_size), 0)
        # self.filtered_scs_softmax_frame /= self.filtered_scs_softmax_frame.sum()
        cropped_chance_nn_integral = local_sum(self.filtered_scs_softmax_frame, self.nn_size)
        # cropped_chance_nn_integral = self.filtered_scs_softmax_frame
        max_change = cropped_chance_nn_integral.max()
        max_index = np.unravel_index(np.argmax(cropped_chance_nn_integral), cropped_chance_nn_integral.shape)
        return max_index, max_change

    def is_in_frame(self, yx):
        # checks if xy is in the frame
        return 0 <= yx[0] < self.frame_size[0] and 0 <= yx[1] < self.frame_size[1]

    def corners_of_crop(self):
        top_left = self.top_left
        bottom_right = self.top_left + self.crop_size
        top_right = [top_left[0], bottom_right[1]]
        bottom_left = [bottom_right[0], top_left[1]]
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def is_crop_in_frame(self):
        # Check if each corner is in the frame
        return all(self.is_in_frame(corner) for corner in self.corners_of_crop())

    def out_of_frame_distance(self):
        # checks the distance from the frame boundaries of each corner of the crop
        return np.array([self.frame_size[0] - self.corners_of_crop()[:, 0], self.frame_size[1] - self.corners_of_crop()[:, 1]]).min()

    def draw_rect_around_cropped_frame(self, frame):
        if self.is_successful:
            cv2.rectangle(frame, tuple(self.top_left[::-1]), tuple(self.bottom_right[::-1]), self.color, 2)

    def draw_cross_on_xy(self, frame):
        if self.is_successful:
            cv2.drawMarker(frame, tuple(self.coordinates.astype(np.int)[::-1]), self.color, cv2.MARKER_CROSS, 20, 2)

    def draw_cross_in_mid_frame(self, frame):
        cv2.drawMarker(frame, tuple(np.array(self.frame_size[:2][::-1]) // 2), (0, 0, 0), cv2.MARKER_CROSS, 20, 6)
        cv2.drawMarker(frame, tuple(np.array(self.frame_size[:2][::-1]) // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    def draw_square_around_xy(self, frame, square_size=None):
        if square_size is None:
            square_size = self.kernel_size
        if self.is_successful:
            square_size = self.kernel_size
            cv2.rectangle(frame, tuple((self.coordinates - square_size // 2).astype(np.int)[::-1]), tuple((self.coordinates + square_size // 2).astype(np.int)[::-1]), self.color, 2)

    def draw_scs_filter_distribution(self, frame, alpha=0.5):
        # draws the filtered_scs_softmax_frame on top of the frame
        if self.is_successful:
            # draw the filtered_scs_softmax_frame
            filtered_scs_softmax_frame = self.filtered_scs_softmax_frame.copy()
            filtered_scs_softmax_frame /= filtered_scs_softmax_frame.max()
            filtered_scs_softmax_frame = (filtered_scs_softmax_frame * 255).astype(np.uint8)
            # _, filtered_scs_softmax_frame = cv2.threshold(filtered_scs_softmax_frame, 158, 255, cv2.THRESH_BINARY)
            filtered_scs_softmax_frame = cv2.applyColorMap(filtered_scs_softmax_frame, cv2.COLORMAP_JET)
            frame[self.top_left[0]:self.bottom_right[0], self.top_left[1]:self.bottom_right[1]] = cv2.addWeighted(frame[self.top_left[0]:self.bottom_right[0], self.top_left[1]:self.bottom_right[1]], alpha, filtered_scs_softmax_frame, 1 - alpha, 0)

    def draw_quadrangle(self, frame):
        if self.is_successful:
            cv2.polylines(frame, [self.kernel_coordinates[:, ::-1]], True, self.color, 2)

    def draw_all_on_frame(self, frame):
        assert self.is_initialized, "SCS_tracker is not initialized. call reset() first."
        # self.draw_scs_filter_distribution(frame)
        self.draw_rect_around_cropped_frame(frame)
        self.draw_square_around_xy(frame)
        self.draw_quadrangle(frame)
        if self.is_successful:
            cv2.putText(frame, str(int(self.max_change)), tuple(self.top_left[::-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 1)
        # self.draw_cross_on_xy(frame)

