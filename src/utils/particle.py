import numpy as np
import cv2

from utils.helper_functions import grid_from_resolution
from utils.helper_functions import softmax, distance_between_pixels


class Particle:
    def __init__(self, kernel_size, crop_size, nn_size, p, q, temperature=1):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.kernel = None
        self.nn_p_avg = np.ones((nn_size, nn_size), np.float32)
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.coordinates_array = np.empty((0, 2), np.int32)
        self.velocity_array = np.empty((0, 2), np.float32)
        self.crop_size = int(crop_size)
        self.p = p
        self.q = q
        self.temperature = temperature
        axis = np.arange(-crop_size // 2, crop_size // 2 + 1)
        X, Y = np.meshgrid(axis, axis)
        # gaussian kernel
        std = 61
        self.nn_p = np.exp(-((X ** 2 + Y ** 2) / (2 * (std // 2) ** 2))) * (1 / (2 * np.pi * (std // 2) ** 2))

    def reset(self):
        self.kernel = None
        self.last_coordinates = None
        self.coordinates = None
        self.coordinates_array = np.empty((0, 2), np.int32)
        self.velocity_array = np.empty((0, 2), np.float32)

    def create_kernel(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        x, y = xy
        if self.kernel is None: # first time
            self.last_coordinates = xy
            self.coordinates_array = np.vstack((self.coordinates_array, np.array(self.last_coordinates).reshape(1, 2)))
            self.velocity = np.zeros(2)
        self.coordinates = xy
        self.kernel = frame[y - self.kernel_size // 2:y + self.kernel_size // 2 + 1, x - self.kernel_size // 2:x + self.kernel_size // 2 + 1].astype(np.float32)
        kernel_norm = np.sqrt(np.square(self.kernel).sum())
        self.kernel = self.kernel / (kernel_norm + 1e-9)

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert self.kernel is not None, "kernel must be created first"
        #crop frame around the last coordinates + velocity:
        x, y = self.coordinates + self.velocity
        x = int(x)
        y = int(y)
        cropped_frame = frame[y - self.crop_size // 2:y + self.crop_size // 2 + 1, x - self.crop_size // 2:x + self.crop_size // 2 + 1]
        if np.any(np.array(cropped_frame.shape) == 0):
            return None, None
        filtered_scs_frame = self.scs_filter(cropped_frame)
        #find the maximum of the filtered_scs_frame
        max_index, max_change = self.find_max(filtered_scs_frame)
        idx = max_index
        # matches = self.find_stochastic_match(filtered_scs_frame)
        # if len(matches) == 0:
        #     print("no matches")
        #     return None, None
        # print("matches.shape" , matches.shape)
        # print("matches", matches)
        # pixels_distance = distance_between_pixels((x, y), matches)
        # min_distance_idx = np.argmin(pixels_distance)
        # min_distance = pixels_distance[min_distance_idx]
        # match = matches[min_distance_idx]
        # match = matches
        # mean, std = self.find_mean(filtered_scs_frame)
        # idx = mean.astype(np.int32)
        #update the coordinates
        self.last_coordinates = self.coordinates
        self.coordinates = np.array(idx) + np.array([x - self.crop_size // 2, y - self.crop_size // 2])
        self.coordinates_array = np.vstack((self.coordinates_array, self.coordinates.reshape(1, 2)))
        #update the velocity
        self.velocity = self.coordinates - self.last_coordinates
        self.velocity_array = np.vstack((self.velocity_array, self.velocity.reshape(1, 2)))
        #create a new kernel
        self.create_kernel(frame, self.coordinates)
        # return self.coordinates
        return self.coordinates, max_change

    def find_stochastic_match(self, filtered_scs_frame, n_samples=10000):
        # match_distribution = softmax(filtered_scs_frame / self.temperature)
        match_distribution = softmax(np.tan(0.99 * np.pi * 0.5 * filtered_scs_frame))
        idx = np.indices(match_distribution.shape).reshape(match_distribution.ndim, -1).T
        # np choose from match_distribution
        samples = np.random.choice(np.arange(len(idx)), replace=True, p=match_distribution.ravel(), size=(n_samples,))
        idx_place, counts = np.unique(samples, return_counts=True)
        choice_idx_img = np.zeros(match_distribution.shape)
        for i, xy in enumerate(idx[idx_place]):
            choice_idx_img[xy[0], xy[1]] = counts[i]
        choice_idx_img = (255 * choice_idx_img / counts.max()).astype(np.uint8)
        # take the top 2 counts:
        top_count_idx = np.argsort(counts)[::-1][0]
        top_idx_place = idx_place[top_count_idx]
        top_idx = idx[top_idx_place]
        # if len(top_2_idx) == 1:
        #     choice_idx = top_2_idx
        # else:
        #     if top_2_counts[0] / top_2_counts[1] > 0.9:
        #         choice_idx = top_2_idx
        #     else:
        #         choice_idx = top_2_idx[0]
        # idx_place = idx_place[counts > 1]
        # choice_idx = idx[idx_place]
        # 2d array sized match_distribution.shape which is filled with ones in choice_idx and 0 elsewhere
        cv2.imshow("choice_idx_img", choice_idx_img)
        cv2.imshow("match_distribution", (255 * match_distribution/match_distribution.max()).astype(np.uint8))
        cv2.waitKey(1)

        # match distribution in index choice_idx:
        # choice_value = np.array([match_distribution[i[0], i[1]] for i in choice_idx])
        # return top k choice_idx by choice_value
        # return choice_idx[np.argsort(choice_value)[::-1]][:top_k]
        return top_idx

    def find_mean(self, filtered_scs_frame):
        match_distribution = softmax(filtered_scs_frame / self.temperature)
        idx = np.indices(match_distribution.shape).reshape(match_distribution.ndim, -1).T.reshape(match_distribution.shape + (match_distribution.ndim,))
        mean = np.sum(idx * match_distribution[..., None], axis=(0, 1))
        std = np.sqrt(np.sum((idx - mean) ** 2 * match_distribution[..., None], axis=(0, 1)))
        return mean, std

    def find_max(self, filtered_scs_frame):
        filtered_scs_softmax_frame = softmax(filtered_scs_frame / self.temperature)
        if filtered_scs_softmax_frame.shape == self.nn_p.shape:
            filtered_scs_softmax_frame *= self.nn_p
            filtered_scs_softmax_frame /= filtered_scs_softmax_frame.sum()
        cropped_chance_nn_integral = cv2.filter2D(filtered_scs_softmax_frame, cv2.CV_32F, self.nn_p_avg)
        # cropped_chance_nn_integral_show = cropped_chance_nn_integral.copy()
        # cropped_chance_nn_integral_show -= cropped_chance_nn_integral_show.min()
        # cropped_chance_nn_integral_show /= (cropped_chance_nn_integral_show.max() + 1e-9)
        # cropped_chance_nn_integral_show = (255 * cropped_chance_nn_integral_show).astype(np.uint8)
        max_change = cropped_chance_nn_integral.max()
        # add max_change as text to cropped_chance_nn_integral_show:
        # cv2.putText(cropped_chance_nn_integral_show, str(100*max_change) + "%", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imshow("filtered_scs_frame", (filtered_scs_softmax_frame / filtered_scs_softmax_frame.max() * 255).astype(np.uint8))
        # cv2.imshow("nn_p", (self.nn_p / self.nn_p.max() * 255).astype(np.uint8))
        cv2.waitKey(1)
        #find the maximum of cropped_chance_nn_integral and return its index as (x, y)
        max_index = np.unravel_index(np.argmax(cropped_chance_nn_integral), cropped_chance_nn_integral.shape)
        # convert to (x, y)
        return max_index[::-1], max_change

    def scs_filter(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert np.any(frame.shape != 0), "frame must not be empty"
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
