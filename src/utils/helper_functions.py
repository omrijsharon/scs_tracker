import numpy as np
import cv2
import json

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum()+1e-15)


def grid_from_resolution(resolution, grid_size, exclude_edges=False):
    # creates grid coordinates such that the screen is divided into grid_size x grid_size squares
    # and the grid is centered on the xy coordinates
    width, height = resolution
    x, y = grid_size
    if exclude_edges:
        x += 2
        y += 2
    # size of each square
    square_width = width // x
    square_height = height // y
    # coordinates of the top left corner of each square
    x_coords = np.arange(width // 2 - square_width * (x // 2), width // 2 + square_width * (x // 2), square_width) + square_width // 2
    y_coords = np.arange(height // 2 - square_height * (y // 2), height // 2 + square_height * (y // 2), square_height) + square_height // 2
    if exclude_edges:
        x_coords = x_coords[1:-1]
        y_coords = y_coords[1:-1]
    return np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)


def distance_between_pixels(pixel, pixel_array):
    # pixel: (x, y)
    # pixel_array: (n, 2)
    return np.sqrt(np.square(pixel_array[:, 0] - pixel[0]) + np.square(pixel_array[:, 1] - pixel[1]))


def argmax2d(array):
    # returns the index of the maximum value in a 2d array
    return np.unravel_index(np.argmax(array), array.shape)


def as_cv_img(img):
    return (img / img.max() * 255).astype(np.uint8)


def normalize_img(img, kernel_size, eps=1e-6):
    k = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return img / (np.sqrt(cv2.filter2D(img.astype(np.float32)**2, cv2.CV_32F, k)) + eps)


def normalize_kernel(k):
    return k / np.linalg.norm(k, keepdims=True)


def frame_to_numpy(frame, height, width):
    img = np.frombuffer(frame.rgb, np.uint8).reshape(height, width, 3)[:, :, ::-1]
    return img.astype(np.uint8)


def particles_mean_std(particles_coordinates, weights=None, mask=None):
    assert not (weights is not None and mask is not None), "weights and mask cannot be both None. If you want to use mask and weights, set weights where mask is false to 0.0"
    if weights is not None:
        assert len(weights) == len(particles_coordinates), "weights and particles_coordinates must have the same length"
        assert np.isclose(np.sum(weights), 1), "weights must sum to 1"
        particles_coordinates = particles_coordinates * weights[:, None]
        mean = np.sum(particles_coordinates, axis=0)
        std = np.sqrt(np.sum(np.square(particles_coordinates - mean) * weights.reshape(-1, 1), axis=0))
        return mean, std
    if mask is not None:
        particles_coordinates = particles_coordinates[mask]
    return np.mean(particles_coordinates, axis=0), np.std(particles_coordinates, axis=0)

def get_particles_attr(particles, attr: str, mask=None):
    values = np.array([getattr(particle, attr) for particle in particles])
    if mask is not None:
        values = values[mask]
    return values


def json_reader(path):
    with open(path, 'r') as f:
        return json.load(f)


def scale_intrinsic_matrix(K, original_frame_size, current_frame_size):
    original_frame_size = np.sort(original_frame_size)
    current_frame_size = np.sort(current_frame_size)
    # Get scale factors
    x_scale = current_frame_size[0] / original_frame_size[0]
    y_scale = current_frame_size[1] / original_frame_size[1]

    # Create a copy of K to modify
    K_scaled = K.copy()

    # Scale fx and cx
    K_scaled[0, 0] *= x_scale  # fx
    K_scaled[0, 2] *= x_scale  # cx

    # Scale fy and cy
    K_scaled[1, 1] *= y_scale  # fy
    K_scaled[1, 2] *= y_scale  # cy

    # K_scaled[2, 2] stays 1

    return K_scaled


# intrinsic matrix if not given:
def create_intrinsic_matrix(width, height, hfov, vfov):
    hfov_rad = np.deg2rad(hfov)
    vfov_rad = np.deg2rad(vfov)
    fx = 0.5 * width / np.tan(0.5 * hfov_rad)
    fy = 0.5 * height / np.tan(0.5 * vfov_rad)
    px = width // 2
    py = height // 2
    K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])  # intrinsic matrix with given parameters
    return K


def match_points(matcher, prev_des, des, min_disparity=8, max_disparity=47, n_neighbors=0):
    if n_neighbors == 0:
        matches = matcher.match(prev_des, des)
        # matches = [m for m in matches if max_disparity > abs(prev_kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) > min_disparity]
        matches = [m for m in matches if min_disparity <= m.distance < max_disparity]
    else:
        matches = matcher.knnMatch(prev_des, des, k=n_neighbors)
        best_matches = []
        for m in matches:
            valid_matches = [match for match in m if min_disparity <= match.distance < max_disparity]
            if valid_matches:
                best_matches.append(min(valid_matches, key=lambda match: match.distance))
        matches = best_matches
    return matches