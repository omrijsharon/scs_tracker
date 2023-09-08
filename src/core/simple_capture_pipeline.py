from core.optical_flow_tracker import OpticalFlowTracker
from utils.helper_functions import get_orb_params_from_trackbars, change_orb_parameters, initialize_orb_trackbars
import utils.screen_capture as sc
import cv2 as cv
import numpy as np
import time

from utils.line_detector import line_detector, corner_detector


def main(tlwh=sc.YOUTUBE_TLWH_SMALL):
    cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)
    window_name = 'SCS Tracker'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    while True:
        frame = cap.capture()
        # line_detector(frame)
        corner_detector(frame)
        # cv.imshow(window_name, gray)
        cv.imshow(window_name, frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
        # time.sleep(1/10)
    cv.destroyAllWindows()
    cap.close()


if __name__ == '__main__':
    main(tlwh=sc.YOUTUBE_TLWH_SMALL)