from rembg import remove
import cv2
from time import perf_counter

time = []

input_path = r'C:\Users\omri_\Downloads\downhill_demo_img.png'
output_path = 'output.png'

input = cv2.imread(input_path)
for _ in range(1):
    start = perf_counter()
    output = remove(input)
    cv2.imshow('output', output[:, :, 1])
    cv2.waitKey(2000)
    time.append(perf_counter() - start)
    print(time[-1])
print(f"Average time: {sum(time)/len(time)}")
cv2.destroyAllWindows()