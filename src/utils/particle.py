import numpy as np
import cv2

from utils.helper_functions import grid_from_resolution
from utils.helper_functions import softmax


class Particle:
    def __init__(self, kernel_size, crop_size, nn_size, p, q, temperature=1):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.kernel = None
        self.nn_p_avg = np.ones((nn_size, nn_size), np.float32)
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.crop_size = int(crop_size)
        self.p = p
        self.q = q
        self.temperature = temperature

    def reset(self):
        self.kernel = None
        self.last_coordinates = None
        self.coordinates = None

    def create_kernel(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        x, y = xy
        if self.kernel is None: # first time
            self.last_coordinates = xy
            self.velocity = np.zeros(2)
        self.coordinates = xy
        self.kernel = frame[y - self.kernel_size // 2:y + self.kernel_size // 2 + 1, x - self.kernel_size // 2:x + self.kernel_size // 2 + 1].astype(np.float32)
        kernel_norm = np.sqrt(np.square(self.kernel).sum())
        self.kernel = self.kernel / (kernel_norm + 1e-9)

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert self.kernel is not None, "kernel must be created first"
        #crop frame around the last coordinates + velocity:
        x, y = self.last_coordinates + self.velocity
        x = int(x)
        y = int(y)
        cropped_frame = frame[y - self.crop_size // 2:y + self.crop_size // 2 + 1, x - self.crop_size // 2:x + self.crop_size // 2 + 1]
        filtered_scs_frame = self.scs_filter(cropped_frame)
        #find the maximum of the filtered_scs_frame
        max_index, max_change = self.find_max(filtered_scs_frame)
        #update the coordinates
        self.last_coordinates = self.coordinates
        self.coordinates = np.array(max_index) + np.array([x - self.crop_size // 2, y - self.crop_size // 2])
        #update the velocity
        self.velocity = self.coordinates - self.last_coordinates
        #create a new kernel
        self.create_kernel(frame, self.coordinates)
        return self.coordinates, max_change

    def find_max(self, filtered_scs_frame):
        filtered_scs_softmax_frame = softmax(filtered_scs_frame / self.temperature)
        cropped_chance_nn_integral = cv2.filter2D(filtered_scs_softmax_frame, cv2.CV_32F, self.nn_p_avg)
        cropped_chance_nn_integral_show = cropped_chance_nn_integral.copy()
        cropped_chance_nn_integral_show -= cropped_chance_nn_integral_show.min()
        cropped_chance_nn_integral_show /= (cropped_chance_nn_integral_show.max() + 1e-9)
        cropped_chance_nn_integral_show = (255 * cropped_chance_nn_integral_show).astype(np.uint8)
        max_change = cropped_chance_nn_integral.max()
        # add max_change as text to cropped_chance_nn_integral_show:
        # cv2.putText(cropped_chance_nn_integral_show, str(100*max_change) + "%", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        # cv2.imshow("filtered_scs_frame", cropped_chance_nn_integral_show)
        # cv2.waitKey(1)
        #find the maximum of cropped_chance_nn_integral and return its index as (x, y)
        max_index = np.unravel_index(np.argmax(cropped_chance_nn_integral), cropped_chance_nn_integral.shape)
        # convert to (x, y)
        return max_index[::-1], max_change

    def scs_filter(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        norm_frame = np.sqrt(cv2.filter2D(frame.astype(np.float32)**2, cv2.CV_32F, self.kernel_ones))
        filtered_frame = cv2.filter2D(frame, cv2.CV_32F, self.kernel)
        filtered_scs_frame = np.sign(filtered_frame) * (np.abs(filtered_frame) / (norm_frame + self.q)) ** self.p
        return filtered_scs_frame


class ParticlesGrid:
    def __init__(self, resolution, kernel_size, crop_size, nn_size, p, q, temperature=1, grid_size=(8, 6)):
        self.grid_size = grid_size
        self.velocities = np.empty((np.prod(grid_size), 2), dtype=np.float32)
        self.grid = grid_from_resolution(resolution, grid_size, exclude_edges=True)
        self.kernel_size = kernel_size
        self.crop_size = crop_size
        self.nn_size = nn_size
        self.p = p
        self.q = q
        self.temperature = temperature
        self.particles = [Particle(kernel_size, crop_size, nn_size, p, q, temperature) for _ in range(grid_size[0] * grid_size[1])]

    def reset(self):
        [p.reset() for p in self.particles]

    def create_kernels(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        # create the kernels from the grid
        [p.create_kernel(frame, xy) for p, xy in zip(self.particles, self.grid)]

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        [p.update(frame) for p in self.particles]
        self.velocities = list(map(lambda p: p.velocity, self.particles))

    def drawParticles(self, frame):
        for i in range(len(self.grid)):
            # draw a circle at each grid point of a particle
            cv2.circle(frame, tuple(self.grid[i]), 1, (0, 0, 255), 1)

    def drawVelocities(self, frame):
        for i in range(len(self.grid)):
            # draw velocity vector arrow from each grid point of a particle
            cv2.arrowedLine(frame, tuple(self.grid[i]), tuple(self.grid[i] + self.velocities[i]), (0, 255, 0), 1)

    def drawKernelsWindows(self, frame):
        # draw kernel's rectangles onto the frame:
        for i in range(len(self.grid)):
            x, y = self.grid[i]
            cv2.rectangle(frame, (x - self.kernel_size // 2, y - self.kernel_size // 2),
                          (x + self.kernel_size // 2, y + self.kernel_size // 2), (255, 0, 0), 1)
