import utils.screen_capture as sc
import cv2 as cv
import numpy as np
import time

class Pipeline:
    def __init__(self, tlwh=sc.YOUTUBE_TLWH_SMALL, window_name='Pipeline'):
        self.cap = sc.ScreenCapture(monitor_number=1, tlwh=tlwh)
        self.window_name = window_name

    # this function should be implemented by the child class.
    def runtime_function(self, frame):
        modifed_frame = frame
        return modifed_frame

    def run(self):
        while True:
            frame = self.cap.capture()
            modifed_frame = self.runtime_function(frame)
            cv.imshow(self.window_name, modifed_frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
        cv.destroyAllWindows()
        self.cap.close()

class ORB_Pipeline(Pipeline):
    def __init__(self, tlwh=sc.YOUTUBE_TLWH_SMALL, window_name='ORB Pipeline'):
        super().__init__(tlwh, window_name)
        self.orb = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.createTrackbar('Max Features', window_name, 10000, 10000, self.f)
        cv.createTrackbar('Scale Factor (x10)', window_name, 15, 40, self.f)
        cv.createTrackbar('Levels', window_name, 8, 20, self.f)
        cv.createTrackbar('WTA_K (2 or 4)', window_name, 2, 4, self.f)
        cv.createTrackbar('edgeThreshold', window_name, 1, 50, self.f)
        cv.createTrackbar('patchSize', window_name, 31, 100, self.f)
        cv.createTrackbar('fastThreshold', window_name, 80, 100, self.f)
        cv.createTrackbar('Max Matches', window_name, 0, 100, self.f)
        cv.createTrackbar('gaussian blur size', window_name, 81, 101, self.f)
        cv.createTrackbar('countor threshold', window_name, 100, 255, self.f)
        cv.createTrackbar('draw keypoints?', window_name, 1, 1, self.f)

    def f(self, x=None):
        return

    def set_orb_params(self):
        self.orb.setMaxFeatures(cv.getTrackbarPos('Max Features', self.window_name))
        self.orb.setScaleFactor(cv.getTrackbarPos('Scale Factor (x10)', self.window_name) / 10.0)
        self.orb.setNLevels(cv.getTrackbarPos('Levels', self.window_name))
        self.orb.setEdgeThreshold(cv.getTrackbarPos('edgeThreshold', self.window_name))
        self.orb.setWTA_K(cv.getTrackbarPos('WTA_K (2 or 4)', self.window_name))
        self.orb.setPatchSize(cv.getTrackbarPos('patchSize', self.window_name))
        self.orb.setFastThreshold(cv.getTrackbarPos('fastThreshold', self.window_name))

    def runtime_function(self, frame):
        self.set_orb_params()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp_density_frame = np.zeros(gray.shape, dtype=np.uint8)
        kp, des = self.orb.detectAndCompute(gray, None)
        if cv.getTrackbarPos('draw keypoints?', self.window_name):
            for k in kp:
                kp_density_frame[int(k.pt[1]), int(k.pt[0])] += 255
            # apply gaussian blur on kp_density_frame
            gaussian_blur_size = cv.getTrackbarPos('gaussian blur size', self.window_name)
            if gaussian_blur_size % 2 == 0:
                gaussian_blur_size += 1
            kp_density_frame = cv.GaussianBlur(kp_density_frame, (gaussian_blur_size, gaussian_blur_size), 0)
            print(kp_density_frame.max())
            kp_density_frame = cv.normalize(kp_density_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
            countor_threshold = cv.getTrackbarPos('countor threshold', self.window_name)
            countor_threshold = countor_threshold if countor_threshold > 0 else 1
            kp_density_frame[kp_density_frame < countor_threshold] = 0
            # kp_density_frame = cv.applyColorMap(kp_density_frame, cv.COLORMAP_JET)
            # contours, hierarchy = cv.findContours(kp_density_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # countours_areas = np.array([cv.contourArea(cnt) for cnt in contours])
            # # sort contours by area
            # contours = [cnt for _, cnt in sorted(zip(countours_areas, contours), key=lambda pair: pair[0])]
            # # find the 2 contours with the largest area and fill its area with 255
            # kp_density_frame = np.zeros(gray.shape[:2], dtype=np.uint8)
            # if len(countours_areas) > 0:
            #     kp_density_frame = cv.drawContours(kp_density_frame, contours[-10:], -1, 255, -1)
            return (0.5 * gray + 0.5 * kp_density_frame).astype(np.uint8)
        return frame


if __name__ == '__main__':
    pipeline = ORB_Pipeline()
    pipeline.run()