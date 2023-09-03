import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self, winSize=15, maxLevel=6):
        self.winSize = (winSize, winSize)
        self.maxLevel = maxLevel
        self.prev_gray = None
        self.frame_size = None
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.is_initialized = False
        self.is_successful = False
        self.color = (0, 255, 0)
        self.prev_gray = None

    def reset(self, frame, xy):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_size = self.prev_gray.shape
        self.velocity = np.zeros(2)
        self.last_coordinates = np.array(xy)
        self.coordinates = np.array(xy)
        self.is_initialized = True
        self.is_successful = True

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_coordinates, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.last_coordinates.reshape(-1, 1, 2).astype(np.float32), None,
                                                              winSize=self.winSize, maxLevel=self.maxLevel)
        if status[0][0] == 1:
            self.last_coordinates = self.coordinates
            self.coordinates = new_coordinates.reshape(2).astype(int)
        else:
            self.is_successful = False
            return False

        self.prev_gray = gray.copy()
        return True

    def is_in_frame(self, xy):
        return 0 <= xy[0] < self.frame_size[0] and 0 <= xy[1] < self.frame_size[1]

    def draw_cross_on_xy(self, frame):
        if self.is_successful:
            cv2.drawMarker(frame, tuple(self.coordinates.astype(int)), self.color, cv2.MARKER_CROSS, 20, 2)

    def draw_all_on_frame(self, frame):
        self.draw_cross_on_xy(frame)