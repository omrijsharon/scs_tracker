from utils.helper_functions import get_orb_params_from_trackbars, change_orb_parameters, initialize_orb_trackbars
import utils.screen_capture as sc
import cv2 as cv
import numpy as np
import time


class SCS_tracker:
    pass


def run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL):
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
            tracker.reset(frame, (x, y))
    tracker = SCS_tracker(n_closest_kp=6)
    cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)

    def f(x=None):
        return

    window_name = 'SCS Tracker'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, mouse_callback)
    # add trackbars for kernel_size=15, crop_size=101, nn_size=5, p=3, q=9, temperature=5, max_velocity=30
    cv.createTrackbar('kernel_size', window_name, 15, 101, f)
    cv.createTrackbar('crop_size', window_name, 101, 301, f)
    cv.createTrackbar('nn_size', window_name, 5, 15, f)
    cv.createTrackbar('p', window_name, 3, 15, f)
    cv.createTrackbar('q', window_name, 9, 15, f)
    cv.createTrackbar('temperature', window_name, 5, 100, f)
    cv.createTrackbar('max_velocity', window_name, 30, 100, f)
    cv.createTrackbar('draw keypoints?', window_name, 1, 1, f)

    while True:
        frame = cap.capture()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if tracker.is_initialized:
            # read trackbars
            # --- Kernel Size ---
            kernel_size = cv.getTrackbarPos('kernel_size', window_name)
            # make kernel_size odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            # make kernel min size 3
            kernel_size = kernel_size if kernel_size >= 3 else 3
            # --- Crop Size ---
            crop_size = cv.getTrackbarPos('crop_size', window_name)
            # make crop_size min size 1.5 * kernel_size + 1
            crop_size = crop_size if crop_size >= 1.5 * kernel_size else int(1.5 * kernel_size)
            # make crop_size odd
            crop_size = crop_size if crop_size % 2 == 1 else crop_size + 1
            # --- nn_size ---
            nn_size = cv.getTrackbarPos('nn_size', window_name)
            # make nn_size odd
            nn_size = nn_size if nn_size % 2 == 1 else nn_size + 1
            # make nn_size min size 3
            nn_size = nn_size if nn_size >= 3 else 3
            # --- p ---
            p = cv.getTrackbarPos('p', window_name)
            # make p min size 1
            p = p if p >= 1 else 1
            # --- q ---
            q = cv.getTrackbarPos('q', window_name)
            # make q min size 1
            q = q if q >= 1 else 1
            q = 10**(-q)
            # --- temperature ---
            temperature = cv.getTrackbarPos('temperature', window_name)
            temperature = temperature if temperature >= 1 else 1
            temperature = temperature / 100
            # --- max_velocity ---
            max_velocity = cv.getTrackbarPos('max_velocity', window_name)




            tracker.update(gray)
            if cv.getTrackbarPos('draw keypoints?', window_name):
                tracker.draw_all_on_frame(frame)
        # cv.imshow(window_name, gray)
        cv.imshow(window_name, frame)
        if cv.waitKey(10) & 0xFF == 27:
            break
        # time.sleep(1/10)
    cv.destroyAllWindows()
    cap.close()


if __name__ == '__main__':
    run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL)