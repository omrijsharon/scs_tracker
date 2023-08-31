import cv2
import numpy as np

from utils.helper_functions import crop_frame, softmax


class SCS_Tracker:
    def __init__(self, kernel_size, crop_size, nn_size, p, q, temperature=1, max_velocity=30):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.kernel = None
        self.frame_size = None
        self.nn_size = nn_size
        self.nn_p_avg = np.ones((nn_size, nn_size), np.float32)
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.top_left = None
        self.filtered_scs_frame = None
        self.filtered_scs_softmax_frame = None
        self.max_velocity = max_velocity  # pixels per frame
        self.crop_size = int(crop_size)
        self.p = p
        self.q = q
        self.temperature = temperature
        axis = np.arange(-crop_size // 2, crop_size // 2 + 1)
        X, Y = np.meshgrid(axis, axis)
        # gaussian kernel
        std = 61
        self.nn_p = np.exp(-((X ** 2 + Y ** 2) / (2 * (std // 2) ** 2))) * (1 / (2 * np.pi * (std // 2) ** 2))
        self.is_initialized = False
        self.is_successful = False
        self.color = (0, 255, 0)

    def change_kernel_size(self, frame, kernel_size):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.create_kernel(frame, self.coordinates)

    def change

    def reset(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        self.frame_size = frame.shape
        self.velocity = np.zeros(2)
        self.create_kernel(frame, xy)
        self.last_coordinates = self.coordinates
        self.is_initialized = True
        self.is_successful = True

    def create_kernel(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        self.coordinates = xy
        self.kernel, self.top_left = crop_frame(frame, xy, self.kernel_size)
        self.kernel /= (np.linalg.norm(self.kernel) + 1e-9)

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert self.kernel is not None, "kernel is None. call reset() first."
        cropped_frame, self.top_left = crop_frame(frame, self.coordinates, self.crop_size)
        if not self.is_in_frame(self.top_left) and not self.is_in_frame(self.top_left + self.crop_size):
            self.is_successful = False
            return False
        # calculate the SCS filter
        self.filtered_scs_frame = self.scs_filter(cropped_frame)
        max_index, max_change = self.find_max(self.filtered_scs_frame)
        idx = max_index
        self.last_coordinates = self.coordinates
        # find the coordinates of the maximum value in the SCS filtered frame
        self.coordinates = idx + self.top_left
        # calculate the velocity
        self.velocity = self.coordinates - self.last_coordinates
        # clip the velocity
        self.velocity = np.clip(self.velocity, a_min=-self.max_velocity, a_max=self.max_velocity)
        # create the kernel
        self.create_kernel(frame, self.coordinates)
        return True

    def scs_filter(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert np.any(frame.shape != 0), "frame must not be empty"
        # norm_frame = np.sqrt(cv2.filter2D(frame.astype(np.float32)**2, cv2.CV_32F, self.kernel_ones))
        frame_square = frame.astype(np.float32)**2
        cv2.boxFilter(frame_square, 0, (self.kernel_size, self.kernel_size), frame_square, (-1, -1), False, cv2.BORDER_DEFAULT)
        norm_frame = np.sqrt(frame_square)
        filtered_frame = cv2.filter2D(frame, cv2.CV_32F, self.kernel)
        filtered_scs_frame = np.sign(filtered_frame) * (np.abs(filtered_frame) / (norm_frame + self.q)) ** self.p
        return filtered_scs_frame

    def find_max(self, filtered_scs_frame):
        self.filtered_scs_softmax_frame = softmax(filtered_scs_frame / self.temperature)
        # apply gaussian blur to filtered_scs_softmax_frame
        self.filtered_scs_softmax_frame = cv2.GaussianBlur(self.filtered_scs_softmax_frame, (self.nn_size, self.nn_size), 0)
        self.filtered_scs_softmax_frame /= self.filtered_scs_softmax_frame.max()
        cropped_chance_nn_integral = cv2.filter2D(self.filtered_scs_softmax_frame, cv2.CV_32F, self.nn_p_avg) # @TODO: Use boxFilter instead of filter2D and get rid of nn_p_avg
        max_change = cropped_chance_nn_integral.max()
        max_index = np.unravel_index(np.argmax(cropped_chance_nn_integral), cropped_chance_nn_integral.shape)
        # convert to (x, y)
        return max_index[::-1], max_change

    def is_in_frame(self, xy):
        # if coordinates are out of image, delete particle
        return 0 <= xy[0] < self.frame_size[1] and 0 <= xy[1] < self.frame_size[0]

    def draw_rect_around_cropped_frame(self, frame):
        if self.is_successful:
            cv2.rectangle(frame, tuple(self.top_left), tuple(self.top_left + self.crop_size), self.color, 2)

    def draw_cross_on_xy(self, frame):
        if self.is_successful:
            cv2.drawMarker(frame, tuple(self.coordinates.astype(np.int)), self.color, cv2.MARKER_CROSS, 20, 2)

    def draw_scs_filter_distribution(self, frame, alpha=0.5):
        # draws the filtered_scs_softmax_frame on top of the frame
        if self.is_successful:
            # draw the filtered_scs_softmax_frame
            filtered_scs_softmax_frame = self.filtered_scs_softmax_frame.copy()
            filtered_scs_softmax_frame /= filtered_scs_softmax_frame.max()
            filtered_scs_softmax_frame = (filtered_scs_softmax_frame * 255).astype(np.uint8)
            filtered_scs_softmax_frame = cv2.applyColorMap(filtered_scs_softmax_frame, cv2.COLORMAP_JET)
            frame[self.top_left[1]:self.top_left[1] + self.crop_size, self.top_left[0]:self.top_left[0] + self.crop_size] = cv2.addWeighted(frame[self.top_left[1]:self.top_left[1] + self.crop_size, self.top_left[0]:self.top_left[0] + self.crop_size], alpha, filtered_scs_softmax_frame, 1 - alpha, 0) # where alpha 0 corresponds to the original image, and alpha 1 corresponds to the filtered_scs_softmax_frame

    def draw_all_on_frame(self, frame):
        assert self.is_initialized, "SCS_tracker is not initialized. call reset() first."
        self.draw_scs_filter_distribution(frame)
        self.draw_rect_around_cropped_frame(frame)
        self.draw_cross_on_xy(frame)
