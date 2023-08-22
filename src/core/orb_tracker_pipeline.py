from utils.orb_tracker import ORB_tracker
import utils.screen_capture as sc
import cv2 as cv
import numpy as np
import time

def run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL):
    def set_orb_params(orb_tracker):
        orb_tracker.set_orb_params(
            maxFeatures=cv.getTrackbarPos('Max Features', window_name),
            scaleFactor=cv.getTrackbarPos('Scale Factor (x10)', window_name) / 10.0,
            nLevels=cv.getTrackbarPos('Levels', window_name),
            WTA_K=cv.getTrackbarPos('WTA_K (2 or 4)', window_name),
            edgeThreshold=cv.getTrackbarPos('edgeThreshold', window_name),
            patchSize=cv.getTrackbarPos('patchSize', window_name),
            fastThreshold=cv.getTrackbarPos('fastThreshold', window_name),
        )
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
            set_orb_params(orb_tracker)
            orb_tracker.reset(frame, (x, y))
    orb_tracker = ORB_tracker(init_crop_size=101)
    cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)

    def f(x=None):
        return

    window_name = 'ORB Tracker'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, mouse_callback)
    cv.createTrackbar('Max Features', window_name, 500, 1000, f)
    cv.createTrackbar('Scale Factor (x10)', window_name, 15, 40, f)
    cv.createTrackbar('Levels', window_name, 8, 20, f)
    cv.createTrackbar('WTA_K (2 or 4)', window_name, 2, 4, f)
    cv.createTrackbar('edgeThreshold', window_name, 1, 50, f)
    cv.createTrackbar('patchSize', window_name, 31, 100, f)
    cv.createTrackbar('fastThreshold', window_name, 72, 100, f)
    cv.createTrackbar('Max Matches', window_name, 0, 100, f)
    cv.createTrackbar('p', window_name, 50, 100, f)
    cv.createTrackbar('draw keypoints?', window_name, 1, 1, f)

    while True:
        frame = cap.capture()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if orb_tracker.is_initialized:
            set_orb_params(orb_tracker)
            # orb_tracker.set_p(cv.getTrackbarPos('p', window_name) / 100.0)
            # orb_tracker.set_max_matches(cv.getTrackbarPos('Max Matches', window_name))
            orb_tracker.update(gray)
            if cv.getTrackbarPos('draw keypoints?', window_name):
                orb_tracker.draw_all_on_frame(frame)
                # orb_tracker.draw_all_on_frame(gray)
            else:
                orb_tracker.draw_xy_on_frame(frame)
        # cv.imshow(window_name, gray)
        cv.imshow(window_name, frame)
        if cv.waitKey(10) & 0xFF == 27:
            break
        # time.sleep(1/10)
    cv.destroyAllWindows()
    cap.close()


if __name__ == '__main__':
    run_orb_tracker(tlwh=sc.YOUTUBE_TLWH_SMALL)