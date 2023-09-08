import cv2
import numpy as np

def corner_detector(frame):
    deg_threshold = 10
    # harris corner detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    gray *= 255.0 / gray.max()
    gray = gray.astype(np.uint8)
    gray = cv2.cornerHarris(gray, 4, 5, 0.06)
    gray = cv2.dilate(gray, np.ones((3, 3), dtype=np.uint8))
    gray *= 255 / (gray.max()/2)
    gray = gray.astype(np.uint8)
    frame *= 0
    frame += gray[..., np.newaxis]
    # frame[gray > 0.11 * gray.max()] = [0, 255, 0]
    gray[gray > 0.11 * gray.max()] = 255
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=10, maxLineGap=180)
    if lines is not None:
        # lines = lines[np.abs(np.arctan2(lines[:, 0, 1] - lines[:, 0, 3], lines[:, 0, 0] - lines[:, 0, 2])) > np.deg2rad(deg_threshold)]
        # lines = lines[np.abs(np.arctan2(lines[:, 0, 1] - lines[:, 0, 3], lines[:, 0, 0] - lines[:, 0, 2])) < np.deg2rad(180 - deg_threshold)]
        # filter lines that their length is more than 1/2 of the frame width
        lines = lines[np.sqrt((lines[:, 0, 0] - lines[:, 0, 2])**2 + (lines[:, 0, 1] - lines[:, 0, 3])**2) < frame.shape[1] / 2]
        # leave only lines that cross each other
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # if the lines are parallel, skip them
                if np.abs(np.arctan2(lines[i, 0, 1] - lines[i, 0, 3], lines[i, 0, 0] - lines[i, 0, 2]) - np.arctan2(lines[j, 0, 1] - lines[j, 0, 3], lines[j, 0, 0] - lines[j, 0, 2])) < np.deg2rad(10):
                    continue
                # if the lines are not parallel, check if they cross each other
                # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
                x1, y1, x2, y2 = lines[i, 0]
                x3, y3, x4, y4 = lines[j, 0]
                denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
                if denominator == 0:
                    continue
                t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denominator
                u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    # the lines cross each other
                    pass
                else:
                    continue
                # if the lines cross each other, draw them
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)


def line_detector(frame):
    deg_threshold = 50
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    # gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    # apply canny edge detection on gray and then apply hough transform on the result. let me have full control over the parameters of both.
    # edges = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
    edges = cv2.Canny(gray, 50, 120, apertureSize=3)
    # dilate the edges
    # edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8))
    # edges = cv2.GaussianBlur(edges, (21, 21), 0)
    frame *= 0
    frame += edges[..., np.newaxis]
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=50, maxLineGap=30)
    # check the angles between the lines and filter out the ones that are smaller than deg_threshold degrees.
    if lines is not None:
        lines = lines[np.abs(np.arctan2(lines[:, 0, 1] - lines[:, 0, 3], lines[:, 0, 0] - lines[:, 0, 2])) > np.deg2rad(deg_threshold)]
        lines = lines[np.abs(np.arctan2(lines[:, 0, 1] - lines[:, 0, 3], lines[:, 0, 0] - lines[:, 0, 2])) < np.deg2rad(180 - deg_threshold)]
        # leave only lines that cross each other
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # if the lines are parallel, skip them
                if np.abs(np.arctan2(lines[i, 0, 1] - lines[i, 0, 3], lines[i, 0, 0] - lines[i, 0, 2]) - np.arctan2(lines[j, 0, 1] - lines[j, 0, 3], lines[j, 0, 0] - lines[j, 0, 2])) < np.deg2rad(10):
                    continue
                # if the lines are not parallel, check if they cross each other
                # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
                x1, y1, x2, y2 = lines[i, 0]
                x3, y3, x4, y4 = lines[j, 0]
                denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
                if denominator == 0:
                    continue
                t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denominator
                u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    # the lines cross each other
                    pass
                else:
                    continue
                # if the lines cross each other, draw them
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # for line in lines:
        #     x1, y1, x2, y2 = line[0] # line is a 2d array with 4 elements. the first two are the coordinates of the first point, the second two are the coordinates of the second point.
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
def line_detector2(frame, ksize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    # create a 2 channel image with the gradient x in the first channel and the gradient y in the second channel and divide each channel by the magnitude so that the magnitude of each pixel is 1.
    grad = np.dstack((grad_x, grad_y))
    grad /= np.dstack((magnitude, magnitude))
    # find how close the nearest pixel is in the direction of the gradient using inner product and filter by magnitude threshold.
