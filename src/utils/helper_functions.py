import numpy as np


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