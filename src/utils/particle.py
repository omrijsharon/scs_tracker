import numpy as np
import cv2

from utils.helper_functions import softmax


class Particle:
    def __init__(self, kernel_size, crop_size, p, q, temperature=1):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.kernel = None
        self.nn_p_avg = np.ones((3, 3), np.float32) / 9
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.crop_size = int(crop_size)
        self.p = p
        self.q = q
        self.temperature = temperature

    def reset(self):
        self.kernel = None
        self.last_coordinates = None
        self.coordinates = None

    def create_kernel(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        x, y = xy
        if self.kernel is None: # first time
            self.last_coordinates = xy
            self.velocity = np.zeros(2)
        self.coordinates = xy
        self.kernel = frame[y - self.kernel_size // 2:y + self.kernel_size // 2 + 1, x - self.kernel_size // 2:x + self.kernel_size // 2 + 1].astype(np.float32)
        kernel_norm = np.sqrt(np.square(self.kernel).sum())
        self.kernel = self.kernel / (kernel_norm + 1e-9)

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert self.kernel is not None, "kernel must be created first"
        #crop frame around the last coordinates + velocity:
        x, y = self.last_coordinates + self.velocity
        x = int(x)
        y = int(y)
        cropped_frame = frame[y - self.crop_size // 2:y + self.crop_size // 2 + 1, x - self.crop_size // 2:x + self.crop_size // 2 + 1]
        filtered_scs_frame = self.scs_filter(cropped_frame)
        #find the maximum of the filtered_scs_frame
        max_index = self.find_max(filtered_scs_frame)
        #update the coordinates
        self.last_coordinates = self.coordinates
        self.coordinates = np.array(max_index) + np.array([x - self.crop_size // 2, y - self.crop_size // 2])
        #update the velocity
        self.velocity = self.coordinates - self.last_coordinates
        #create a new kernel
        self.create_kernel(frame, self.coordinates)
        return self.coordinates

    def find_max(self, filtered_scs_frame):
        filtered_scs_softmax_frame = softmax(filtered_scs_frame / self.temperature)
        cropped_chance_nn_integral = cv2.filter2D(filtered_scs_softmax_frame, cv2.CV_32F, self.nn_p_avg)
        cropped_chance_nn_integral_show = cropped_chance_nn_integral.copy()
        cropped_chance_nn_integral_show -= cropped_chance_nn_integral_show.min()
        cropped_chance_nn_integral_show /= (cropped_chance_nn_integral_show.max() + 1e-9)
        cv2.imshow("filtered_scs_frame", (255 * cropped_chance_nn_integral_show).astype(np.uint8))
        cv2.waitKey(1)
        max_index = np.unravel_index(np.argmax(cropped_chance_nn_integral), cropped_chance_nn_integral.shape)
        return max_index

    def scs_filter(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        norm_frame = np.sqrt(cv2.filter2D(frame.astype(np.float32)**2, cv2.CV_32F, self.kernel_ones))
        filtered_frame = cv2.filter2D(frame, cv2.CV_32F, self.kernel)
        filtered_scs_frame = np.sign(filtered_frame) * (np.abs(filtered_frame) / (norm_frame + self.q)) ** self.p
        return filtered_scs_frame

