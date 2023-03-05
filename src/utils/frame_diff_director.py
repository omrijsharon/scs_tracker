import cv2
import numpy as np

from utils.scs import scs_filter

class Director:
    # this class is initiated with a blank frame, the size of the captured frame.
    # it is used to calculate the difference between the current frame and the previous frame.
    # then, using a gradient kernel sized kernel_size, and its 90 degrees rotated version, it filteres the difference image.
    # the filtered images are used to calculate the angle and the magnitude of the gradient and thus the direction of the movement.
    def __init__(self,kernel_size, w, h, p, q, gradient_func: str="sin"):
        """
        :param kernel_size: size of the gradient kernel
        :param w: width of the frame
        :param h: height of the frame
        :param p: the power of the SCS filter
        :param q: the floor noise of the SCS filter
        :param gradient_func: the function used to generate the gradient kernel. can be "sin", "tanh" of "linear"
        """
        self.p = p
        self.q = q
        self.kernel_size = kernel_size
        self.previous_frame = np.zeros((h, w), dtype=np.float32)
        # generate gradient kernels by kernel size:
        self.kernel = np.zeros((kernel_size, kernel_size, 2), dtype=np.float32)
        axis = np.linspace(-1, 1, kernel_size)
        x, y = np.meshgrid(axis, axis)
        if gradient_func == "sin":
            self.kernel[:, :, 0] = np.sin(np.pi/2 * x)
            self.kernel[:, :, 1] = np.sin(np.pi/2 * y)
        elif gradient_func == "tanh":
            self.kernel[:, :, 0] = np.tanh(np.pi/2 * x)
            self.kernel[:, :, 1] = np.tanh(np.pi/2 * y)
        elif gradient_func == "linear":
            self.kernel[:, :, 0] = x
            self.kernel[:, :, 1] = y
        else:
            raise ValueError("gradient_func must be 'sin', 'tanh' or 'linear'")
        self.kernel = self.kernel / np.linalg.norm(self.kernel, axis=(0, 1), keepdims=True)
        self.ones_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        self.filtered_frame = np.zeros((h, w, 2), dtype=np.float32)
        self.filtered_scs_frame = np.zeros((h, w, 2), dtype=np.float32)

    def calculate(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        assert frame.dtype == 'float32', "frame must be np.float32"
        assert frame.shape == self.previous_frame.shape, "frame must be the same size as the previous frame"
        # frame must be between -1 and 1 for the SCS filter to work properly
        assert frame.max() <= 1 and frame.min() >= -1, "frame must be between -1 and 1"
        # calculate the difference between the current frame and the previous frame
        # diff = (frame - self.previous_frame) / (2 * self.kernel_size)
        diff = frame / self.kernel_size
        # diff /= np.prod(diff.shape)
        #filter the difference image with ones_kernel:
        # norm_frame = np.sqrt(cv2.filter2D(frame ** 2, cv2.CV_32F, self.ones_kernel))
        # filter the difference image with the gradient kernels
        for i in range(self.kernel.shape[2]):
            self.filtered_frame[:, :, i] += cv2.filter2D(diff, cv2.CV_32F, self.kernel[:, :, i])
            # self.filtered_scs_frame[:, :, i] += np.sign(self.filtered_frame[:, :, i]) * (np.abs(self.filtered_frame[:, :, i]) / (norm_frame + self.q)) ** self.p
        # calculate the angle and the magnitude of the gradient
        angle = np.arctan2(self.filtered_frame[:, :, 1], self.filtered_frame[:, :, 0])
        magnitude = np.sqrt((self.filtered_frame**2).sum(axis=2))
        # calculate the direction of the movement
        # direction = np.arctan2(np.sin(angle), np.cos(angle))
        # update the previous frame
        self.previous_frame = 0
        self.previous_frame += frame
        self.filtered_frame = 0
        self.filtered_scs_frame = 0
        return angle, magnitude

    def hsv_projection(self, angle, magnitude):
        # this function is used to visualize the direction and the magnitude of the gradient.
        # the direction is visualized by the hue of the image, and the magnitude is visualized by the saturation of the image.
        # the value of the image is always 1.
        hsv = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.float32)
        hsv[:, :, 0] = (np.rad2deg(angle % (2*np.pi))).astype(np.uint8)
        hsv[:, :, 1] = 1
        hsv[:, :, 2] = (magnitude * 8).astype(np.uint8)
        return hsv