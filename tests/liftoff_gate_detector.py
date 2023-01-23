import cv2
import mss
import numpy as np

from core.winfo import getWindowSizes

from utils.helper_functions import softmax, argmax2d, as_cv_img, normalize_img, normalize_kernel, frame_to_numpy


if __name__ == '__main__':
    win_sizes = getWindowSizes()
    liftoff_winfo = [d for d in win_sizes if d['name']=="Liftoff"]
    assert len(liftoff_winfo) == 1, "Liftoff window not found. If open, please bring it to the foreground."
    liftoff_winfo = liftoff_winfo[0]
    shave_pixels = {"top": 31, "left": 8, "width": 16, "height": 39}
    kernel_size = 23
    #import image
    mark_liftoff = cv2.imread(r'C:\Users\omri_\OneDrive\Documents\fpv\liftoff\mark.png')
    mark_liftoff = cv2.cvtColor(mark_liftoff, cv2.COLOR_BGR2GRAY)
    mark_liftoff = normalize_kernel(mark_liftoff)
    screenshot = cv2.imread(r'C:\Users\omri_\OneDrive\Documents\fpv\liftoff\screenshot.png')
    mark = np.zeros((kernel_size, kernel_size), np.uint8)
    cv2.circle(mark, (11, 11), 8, 255, 3)
    mark = normalize_kernel(mark)
    gray_screenshot = normalize_img(screenshot[:, :, 2], kernel_size, eps=1e-9)
    # gray_screenshot = normalize_img(cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY), kernel_size, eps=1e-9)
    # conv = cv2.filter2D(gray_screenshot, cv2.CV_32F, mark_liftoff)
    conv = cv2.filter2D(gray_screenshot, cv2.CV_32F, mark)
    conv_softmax = softmax(conv / 0.2)
    conv_softmax_show = as_cv_img(conv_softmax)
    idx = argmax2d(conv)
    conv_show = (conv / conv.max() * 255).astype(np.uint8)
    cv2.circle(conv_show, idx[::-1], 20, 255, 5)
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
        while True:
            img_byte = sct.grab(monitor)
            img = frame_to_numpy(img_byte, monitor["height"], monitor["width"])[:, :, ::-1].astype(np.uint8)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
