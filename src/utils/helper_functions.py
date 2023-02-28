import numpy as np
import cv2


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


def particles_mean_std(particles_coordinates, mask=None):
    if mask is not None:
        particles_coordinates = particles_coordinates[mask]
    return np.mean(particles_coordinates, axis=0), np.std(particles_coordinates, axis=0)


def get_particles_coordinates(particles):
    return np.array([particle.coordinates for i, particle in enumerate(particles)])
