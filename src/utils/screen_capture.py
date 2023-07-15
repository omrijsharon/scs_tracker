import cv2
import numpy as np
import mss

YOUTUBE_TLWH_SMALL = (160, 2019, 1280, 720)
YOUTUBE_TLWH_LARGE = (80, 1921, 1901, 1135)

class ScreenCapture:
    def __init__(self, monitor_number=1, tlwh=YOUTUBE_TLWH_SMALL):
        self.sct = mss.mss()
        mon = self.sct.monitors[monitor_number]
        self.monitor = {
            "top": mon["top"] + tlwh[0],  # 100px from the top
            "left": mon["left"] + tlwh[1],  # 100px from the left
            "width": tlwh[2],
            "height": tlwh[3],
            "mon": monitor_number,
        }

    def capture(self):
        # Capture the screenshot
        img_byte = self.sct.grab(self.monitor)
        # Convert the raw bytes into a numpy array
        img = np.array(img_byte)
        # Convert the array into an OpenCV image
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def close(self):
        self.sct.close()