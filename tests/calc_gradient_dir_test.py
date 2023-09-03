import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils.helper_functions import calc_gradient_dir
import utils.screen_capture as sc

kernel_size = 21
tlwh=sc.YOUTUBE_TLWH_SMALL
cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)
frame = cap.capture()
# convert frame to grayscale
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
kernel_gradient_dir = None
x_mouse, y_mouse = None, None

# plt.figure(figsize=(8, 8))
# # ax = plt.subplot(111, projection='polar')
# ax = plt.subplot(111)

def create_kernel(frame, xy, kernel_size):
    x, y = xy
    kernel = frame[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1].astype(np.float32)
    return kernel


# opencv for showing the captured frame and get x y mouse click for the kernel
def mouse_callback(event, x, y, flags, param):
    global kernel_gradient_dir, x_mouse, y_mouse
    if event == cv.EVENT_LBUTTONDOWN:
        frame = cap.capture()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_mouse, y_mouse = x, y
        print("mouse clicked at:", x, y)
        kernel = create_kernel(frame, (x, y), kernel_size)
        kernel_gradient_dir = calc_gradient_dir(kernel, kernel_size)


window_name = 'SCS Tracker'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.setMouseCallback(window_name, mouse_callback)

while True:
    frame = cap.capture()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if kernel_gradient_dir is not None:
        kernel_gradient_dir_angle = (np.rad2deg(np.arctan2(kernel_gradient_dir[:, :, 1], kernel_gradient_dir[:, :, 0])).flatten())
        kernel_gradient_dir_flatten = kernel_gradient_dir.reshape(-1, 2)
        # add arrows in x_mouse, y_mouse position pointing to the direction of the kernel's gradients with alpha 0.05
        for i in range(len(kernel_gradient_dir_flatten)):
            cv.arrowedLine(frame, (x_mouse, y_mouse), (x_mouse + int(kernel_gradient_dir_flatten[i, 0] * 100), y_mouse + int(kernel_gradient_dir_flatten[i, 1] * 100)), (255, 0, 0), 1, tipLength=0.1)
        # clear the plot
        plt.clf()
        # plt.hist(kernel_gradient_dir_angle, bins=36, range=(-180, 180), align='mid', alpha=0.3)
        bins = np.linspace(-180, 180, 32, endpoint=False)
        hist, bin_edges = np.histogram(kernel_gradient_dir_angle, bins=bins)
        # find in what bin the histogram count is the highest and translate the bin to degrees
        max_bin = np.argmax(hist)
        max_bin_degrees = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
        # draw an arrow in the direction of max_bin_degrees
        cv.arrowedLine(frame, (x_mouse, y_mouse), (x_mouse + int(np.cos(np.deg2rad(max_bin_degrees)) * 100), y_mouse + int(np.sin(np.deg2rad(max_bin_degrees)) * 100)), (0, 255, 0), 1, tipLength=0.1)
        # plt.bar((bin_edges[:-1]-max_bin_degrees), hist, width=10, align='edge', alpha=0.6)
        plt.bar((bin_edges[:-1]), hist, width=10, align='edge', alpha=0.6)
        plt.pause(0.01)

    cv.imshow(window_name, frame)
    if cv.waitKey(10) & 0xFF == 27:
        break
cv.destroyAllWindows()
cap.close()




