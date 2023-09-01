import cv2 as cv
import numpy as np

# this script tests if boxFilter and cv2.filter2D are equivalent if filter2D is used with a kernel of ones

# create a random image
frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
boxFrame = np.zeros_like(frame, dtype=np.float32)
# create a kernel
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
# filter the image with boxFilter
filtered_frame1 = cv.boxFilter(frame, cv.CV_32F, (kernel_size, kernel_size), normalize=False)
# filter the image with filter2D
filtered_frame2 = cv.filter2D(frame, cv.CV_32F, kernel)
# compare the results
print("boxFilter and filter2D are equal:", np.allclose(filtered_frame1, filtered_frame2))
# show frame difference
cv.imshow("frame difference", np.abs(filtered_frame1 - filtered_frame2).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()
