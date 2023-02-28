import numpy as np
import cv2
from rembg import remove, new_session

from utils.helper_functions import grid_from_resolution
from utils.helper_functions import softmax, distance_between_pixels


class Particle:
    def __init__(self, kernel_size, crop_size, nn_size, p, q, temperature=1, pixel_noise=1, rembg=False):
        self.kernel_size = kernel_size
        self.kernel_ones = np.ones((kernel_size, kernel_size), np.float32)
        self.kernel = None
        self.kernel_rot90 = None
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
        self.pixel_noise = int(pixel_noise)
        self.rembg = rembg

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
        # rotated kernel in 90 degrees:
        # self.kernel_rot90 = np.rot90(self.kernel, 1)


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
        if self.rembg:
            output = remove((((cropped_frame+1) / 2) * 255).astype(np.uint8), alpha_matting=True)
            cv2.imshow("output", output)
            cv2.waitKey(1)
        filtered_scs_frame = self.scs_filter(cropped_frame)
        # filtered_scs_frame = self.scs_filter_angle_invariant(cropped_frame)
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
        # add pixel noise to the coordinates
        self.coordinates = self.coordinates + np.random.randint(-self.pixel_noise, self.pixel_noise + 1, 2)
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
        # cv2.imshow("filtered_scs_frame", (filtered_scs_softmax_frame / filtered_scs_softmax_frame.max() * 255).astype(np.uint8))
        # cv2.imshow("nn_p", (self.nn_p / self.nn_p.max() * 255).astype(np.uint8))
        # cv2.waitKey(1)
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

    def scs_filter_angle_invariant(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert np.any(frame.shape != 0), "frame must not be empty"
        norm_frame = np.sqrt(cv2.filter2D(frame.astype(np.float32)**2, cv2.CV_32F, self.kernel_ones))
        filtered_frame = cv2.filter2D(frame, cv2.CV_32F, self.kernel)
        filtered_scs_frame = np.sign(filtered_frame) * (np.abs(filtered_frame) / (norm_frame + self.q)) ** self.p
        filtered_frame_rot90 = cv2.filter2D(frame, cv2.CV_32F, self.kernel_rot90)
        filtered_scs_frame_rot90 = np.sign(filtered_frame_rot90) * (np.abs(filtered_frame_rot90) / (norm_frame + self.q)) ** self.p
        return np.sqrt(((filtered_scs_frame+1)/2) ** 2 + ((filtered_scs_frame_rot90+1)/2) ** 2) - 1

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


class ParticleRembg:
    def __init__(self, max_crop_size: int = 250):
        # self.session = new_session(model_name="silueta")
        self.session = new_session()
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = None
        self.coordinates_array = np.empty((0, 2), np.int32)
        self.velocity_array = np.empty((0, 2), np.float32)
        self.max_crop_size = max_crop_size
        self.crop_size = np.array([self.max_crop_size, self.max_crop_size])
        self.crop = np.zeros((self.crop_size[1], self.crop_size[0]), np.uint8)
        self.canny_top_percent = 0.8
        self.canny_bottom_percent = 0.2
        self.bbox_params = None

    def reset(self, frame):
        self.last_coordinates = None
        self.coordinates = None
        self.velocity = np.zeros(2)
        self.coordinates_array = np.empty((0, 2), np.int32)
        self.velocity_array = np.empty((0, 2), np.float32)
        cropped_rembg = remove(frame, alpha_matting=True)
        x, y, w, h = self.find_bounding_box(cropped_rembg)
        self.bbox_params = (x, y, w, h)
        if x is None:
            raise ValueError("No object found in the frame")
        # self.show_boundingbox(frame, x, y, w, h)
        self.coordinates = np.array([x + w // 2, y + h // 2])
        self.crop_size = np.array([min(w, self.max_crop_size), min(h, self.max_crop_size)])
        self.create_crop(frame, self.coordinates)
        self.coordinates_array = np.vstack((self.coordinates_array, self.coordinates.reshape(1, 2)))
        # self.show_crop(frame)

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
        # rotated kernel in 90 degrees:
        # self.kernel_rot90 = np.rot90(self.kernel, 1)

    def update(self, frame):
        assert frame.ndim == 2, "frame must be grayscale"
        assert frame.dtype == np.uint8, "frame must be np.uint8"
        self.last_coordinates = self.coordinates
        cropped_rembg = remove(self.crop, session=self.session, alpha_matting=True)
        certainty = cropped_rembg.max()/255
        # print("certainty: ", certainty * 100 , "%")
        x, y, w, h = self.find_bounding_box(cropped_rembg)
        if x is None:
            raise ValueError("No object found in the frame")
        self.coordinates = np.array([x + w//2, y + h//2]) + np.array([self.coordinates[0] - self.crop_size[0] // 2, self.coordinates[1] - self.crop_size[1] // 2])
        self.bbox_params = (*(np.array([x, y]) + np.array([self.coordinates[0] - self.crop_size[0] // 2, self.coordinates[1] - self.crop_size[1] // 2])), w, h)
        # Why did the velocity change?
        self.velocity = self.coordinates - self.last_coordinates
        # self.show_boundingbox(frame, *self.coordinates, w, h)
        self.crop_size = np.array([min(w, self.max_crop_size), min(h, self.max_crop_size)])
        #crop frame around the last coordinates + velocity:
        estimated_coordinates = self.coordinates + self.velocity*0
        self.create_crop(frame, estimated_coordinates)
        if np.any(np.array(self.crop.shape) == 0):
            return None, None
        # cv2.imshow("cropped_rembg", cropped_rembg)
        # cv2.waitKey(1)
        if w == 0 or h == 0:
            return None, None
        self.coordinates_array = np.vstack((self.coordinates_array, self.coordinates.reshape(1, 2)))
        self.velocity_array = np.vstack((self.velocity_array, self.velocity.reshape(1, 2)))
        return self.coordinates, certainty

    def show_crop(self, frame):
        x, y = (self.coordinates-self.crop_size//2).astype(np.int32)
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x, y), (x + self.crop_size[0], y + self.crop_size[1]), (0, 255, 0), 2)
        cv2.imshow("cropped_frame", frame_copy)
        cv2.waitKey(1)

    def show_boundingbox(self, frame, x, y, w, h):
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("frame with boundingbox", frame_copy)
        cv2.waitKey(1)


    def find_bounding_box(self, cropped_rembg):
        # find the bounding box of the object in the cropped frame
        cropped_rembg = cv2.cvtColor(cropped_rembg, cv2.COLOR_BGR2GRAY)
        cv2.imshow("cropped_rembg", cropped_rembg)
        cv2.waitKey(1)
        cropped_rembg = cv2.GaussianBlur(cropped_rembg, (5, 5), 0)
        cropped_rembg_max = cropped_rembg.max()
        canny_lower_threshold = max(int(cropped_rembg_max * self.canny_bottom_percent), 10)
        canny_higher_threshold = min(int(cropped_rembg_max * self.canny_top_percent), 240)
        print("canny thresholds", canny_lower_threshold, canny_higher_threshold)
        cropped_rembg = cv2.Canny(cropped_rembg, canny_lower_threshold, canny_higher_threshold)
        # cv2.imshow("Canny_rembg", cropped_rembg)
        # cv2.waitKey(1)
        # contours, _ = cv2.findContours(cropped_rembg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros(cropped_rembg.shape, np.uint8)
        # for cnt in contours:
        #     cv2.drawContours(mask, [cnt], 0, 255, 1)
        cropped_rembg = cv2.dilate(cropped_rembg, None, iterations=5)
        cropped_rembg = cv2.erode(cropped_rembg, None, iterations=5)
        # cv2.imshow("cropped_rembg", cropped_rembg)
        # cv2.waitKey(1)
        # contours, _ = cv2.findContours(cropped_rembg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(cropped_rembg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 4 * [None]
        # find the bounding box of the contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h

    def create_crop(self, frame, xy):
        assert frame.ndim == 2, "frame must be grayscale"
        x, y = xy
        self.coordinates = np.array(xy)
        x = int(x)
        y = int(y)
        self.crop = frame[y - self.crop_size[1] // 2:y + self.crop_size[1] // 2 + 1, x - self.crop_size[0] // 2:x + self.crop_size[0] // 2 + 1]