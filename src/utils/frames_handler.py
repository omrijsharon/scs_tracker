import numpy as np
import mss
import cv2
from functools import partial

from utils.scs import create_kernel
from utils.image_proc import image_processing
from utils.helper_functions import softmax

SMALL_TOP = 160
SMALL_LEFT = 2019
SMALL_WIDTH = 1280
SMALL_HEIGHT = 720
LARGE_TOP = 80
LARGE_LEFT = 1921
LARGE_WIDTH = 1901
LARGE_HEIGHT = 1135
youtube_tlwh_small = (160, 2019, 1280, 720)
youtube_tlwh_large = (80, 1921, 1901, 1135)

mouse_coords = (-1, -1)
kernel_size = 51
kernel = np.ones((kernel_size, kernel_size), np.float32)
is_kernel = False


def capture_frames_live(img_proc_func, tlwh=None, monitor_number=0):
    global mouse_coords, kernel, is_kernel
    def mouse_callback(event, x, y, flags, param):
        global mouse_coords, kernel, is_kernel
        kernel_size = param
        if event == cv2.EVENT_LBUTTONUP:
            is_kernel = False
            mouse_coords = (x, y)
            kernel = create_kernel(img, mouse_coords, kernel_size)
            is_kernel = True

    if tlwh is None:
        tlwh = (SMALL_TOP, SMALL_LEFT, SMALL_WIDTH, SMALL_HEIGHT)
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        # The screen part to capture
        monitor = {
            "top": mon["top"] + tlwh[0],  # 100px from the top
            "left": mon["left"] + tlwh[1],  # 100px from the left
            "width": tlwh[2],
            "height": tlwh[3],
            "mon": monitor_number,
        }
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', mouse_callback, kernel_size)
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, tlwh[3], tlwh[2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if is_kernel:
                # cv2.imshow("kernel", (255 * kernel/kernel.max()).astype(np.uint8))
                filtered_scs_img = img_proc_func(img, kernel)
                # find the max x,y indices of the filtered_scs_img:
                max_y, max_x = np.unravel_index(filtered_scs_img.argmax(), filtered_scs_img.shape)
                filtered_img_show = softmax(filtered_scs_img)
                filtered_img_show = (255 * filtered_img_show / filtered_img_show.max()).astype(np.uint8)
                filtered_img_show = cv2.cvtColor(filtered_img_show, cv2.COLOR_GRAY2BGR)
                cv2.circle(filtered_img_show, mouse_coords, kernel_size//2, (0, 255, 0), 2)
                cv2.circle(filtered_img_show, (max_x, max_y), kernel_size//3, (0, 0, 255), 2)
                cv2.imshow('scs', filtered_img_show)
                mouse_coords = max_x, max_y
                kernel = create_kernel(img, mouse_coords, kernel_size)
            img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if mouse_coords[0] > -1:
                cv2.circle(img_show, mouse_coords, kernel_size//2, (0, 255, 0), 2)
                cv2.imshow('frame', img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def frame_to_numpy(frame, height, width):
    img = np.frombuffer(frame.rgb, np.uint8).reshape(height, width, 3)[:, :, ::-1]
    return img.astype(np.uint8)

if __name__ == '__main__':
    img_proc = partial(image_processing, ones_kernel=np.ones((kernel_size, kernel_size), np.float32), p=3, q=1e-9)
    capture_frames_live(img_proc, youtube_tlwh_small)