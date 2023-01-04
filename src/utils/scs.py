import cv2
import numpy as np


def create_kernel(frame, xy, kernel_size):
    x, y = xy
    # ones_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # crop a kernel sized "kernel_size" from the frame with the center at (x, y)
    kernel = frame[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1].astype(np.float32)
    # convert grayscale to BGR
    # cv2.imshow("kernel", cv2.cvtColor(kernel, cv2.COLOR_GRAY2BGR))
    # cv2.waitKey(1)
    kernel_norm = np.sqrt(np.square(kernel).sum())
    return kernel / (kernel_norm + 1e-6)


def scs_filter(frame, kernel, ones_kernel, p=3, q=1e-6):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm_frame = np.sqrt(cv2.filter2D(frame.astype(np.float32)**2, cv2.CV_32F, ones_kernel))
    filted_frame = cv2.filter2D(frame, cv2.CV_32F, kernel)
    filted_scs_frame = np.sign(filted_frame) * (np.abs(filted_frame) / (norm_frame + q)) ** p
    return filted_scs_frame