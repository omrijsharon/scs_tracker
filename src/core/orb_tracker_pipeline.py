from utils.helper_functions import get_orb_params_from_trackbars, change_orb_parameters, initialize_orb_trackbars
from utils.orb_tracker import ORB_tracker
import utils.screen_capture as sc
import cv2 as cv
import numpy as np
import time

def run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL):
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
            change_orb_parameters(orb_tracker.orb, **get_orb_params_from_trackbars(window_name))
            orb_tracker.reset(frame, (x, y))
    orb_tracker = ORB_tracker(init_crop_size=101)
    cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)

    def f(x=None):
        return

    window_name = 'ORB Tracker'
    initialize_orb_trackbars(window_name, callback_func=mouse_callback)
    cv.createTrackbar('Max Matches', window_name, 0, 100, f)
    cv.createTrackbar('p', window_name, 50, 100, f)
    cv.createTrackbar('draw keypoints?', window_name, 1, 1, f)

    while True:
        frame = cap.capture()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if orb_tracker.is_initialized:
            change_orb_parameters(orb_tracker.orb, **get_orb_params_from_trackbars(window_name))
            # orb_tracker.set_p(cv.getTrackbarPos('p', window_name) / 100.0)
            # orb_tracker.set_max_matches(cv.getTrackbarPos('Max Matches', window_name))
            orb_tracker.update(gray)
            if cv.getTrackbarPos('draw keypoints?', window_name):
                orb_tracker.draw_all_on_frame(frame)
                # orb_tracker.draw_all_on_frame(gray)
            else:
                orb_tracker.draw_xy_on_frame(frame)
                orb_tracker.draw_cross_on_xy_follow(frame, radius=10, thickness=2, cross_size=10, outer_radius=5)
        # cv.imshow(window_name, gray)
        cv.imshow(window_name, frame)
        if cv.waitKey(10) & 0xFF == 27:
            break
        # time.sleep(1/10)
    cv.destroyAllWindows()
    cap.close()


if __name__ == '__main__':
    run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL)