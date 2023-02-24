import cv2
import mss
import numpy as np

from core.winfo import getWindowSizes

from utils.helper_functions import softmax, argmax2d, as_cv_img, normalize_img, normalize_kernel, frame_to_numpy


def detect_gate_copilot(frame):
    frame = frame_to_numpy(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = normalize_img(frame)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.Canny(frame, 100, 200)
    frame = cv2.dilate(frame, np.ones((3, 3), np.uint8), iterations=1)
    frame = cv2.erode(frame, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], 0, (255, 0, 0), 5)
            break
    return frame


def detect_gate(normalized_frame, gate_mark):
    conv = cv2.filter2D(normalized_frame, cv2.CV_32F, gate_mark)
    return argmax2d(conv)


if __name__ == '__main__':
    win_sizes = getWindowSizes()
    liftoff_winfo = [d for d in win_sizes if d['name']=="Liftoff"]
    assert len(liftoff_winfo) == 1, "Liftoff window not found. If open, please bring it to the foreground."
    liftoff_winfo = liftoff_winfo[0]
    shave_pixels = {"top": 31, "left": 8, "width": 16, "height": 39}
    kernel_size = 16

    gate_mark = np.zeros((kernel_size, kernel_size), np.uint8)
    cv2.circle(gate_mark, (kernel_size // 2 - 1, kernel_size // 2 - 1), 6, 255, 3)
    gate_mark = normalize_kernel(gate_mark)

    horizon_mark = np.zeros((12, 12))
    # draw rectangle sized 6x6 in the middle of the horizon_mark
    cv2.rectangle(horizon_mark, (3, 3), (8, 8), 1, -1)
    horizon_mark = normalize_kernel(horizon_mark)

    monitor_number = 0
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        # The screen part to capture
        monitor = {
            "top": liftoff_winfo['rect'][1]+shave_pixels['top'],
            "left": liftoff_winfo['rect'][0]+shave_pixels['left'],
            "width": liftoff_winfo['width']-shave_pixels['width'],
            "height": liftoff_winfo['height']-shave_pixels['height'],
            "mon": monitor_number,
        }
        cv2.namedWindow('frame')
        i=0
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, monitor["height"], monitor["width"])
            gray_screenshot = normalize_img(img[:, :, 2], kernel_size, eps=1e-9)
            idx = detect_gate(gray_screenshot, gate_mark)
            if i%10==0:
                cv2.circle(img, idx[::-1], 20, 255, 5)
                cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
