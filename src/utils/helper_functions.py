import numpy as np
import cv2 as cv
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
    return img / (np.sqrt(cv.filter2D(img.astype(np.float32)**2, cv.CV_32F, k)) + eps)


def normalize_kernel(k):
    return k / np.linalg.norm(k, keepdims=True)


def local_sum(frame, kernel_size):
    return cv.boxFilter(frame, cv.CV_32F, (kernel_size, kernel_size), normalize=False)


def local_magnitude(frame, kernel_size):
    # assuming square kernel
    return np.sqrt(local_sum(frame.astype(np.float32) ** 2, kernel_size))


def calc_scs(frame, kernel, p=3, q=1e-6):
    # assuming square kernel
    filtered_frame = cv.filter2D(frame, cv.CV_32F, kernel)
    return np.sign(filtered_frame) * (np.abs(filtered_frame) / (local_magnitude(frame, kernel.shape[0]) + q)) ** p


def calc_scs_multi_channels(frame, kernel, p=3, q=1e-6):
    conv_frame = filter2d_multi_channels(frame, kernel)
    local_mag = local_magnitude_multi_channels(frame, kernel.shape[0])
    return np.sign(conv_frame) * (np.abs(conv_frame) / (local_mag + q)) ** p


def scs_matcher(cropped_frame, kernel, top_left, p=3, q=1e-6):
    scs = calc_scs(cropped_frame, kernel, p, q)
    return top_left + argmax2d(scs)


def scs_matcher_pool(args):
    cropped_frame, kernel, top_left, p, q = args
    scs = calc_scs(cropped_frame, kernel, p, q)
    return top_left + argmax2d(scs)


def filter2d_multi_channels(frame, kernel):
    result = np.zeros_like(frame)
    for i in range(frame.shape[2]):
        result[:, :, i] = cv.filter2D(frame[:, :, i], cv.CV_32F, kernel)
    return result.sum(axis=2)


def local_magnitude_multi_channels(frame, kernel_size):
    return np.sqrt(local_sum(frame, kernel_size).sum(axis=2))


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


# def match_points(matcher, prev_des, des, min_disparity=8, max_disparity=47, n_neighbors=0):
#     if n_neighbors == 0:
#         matches = matcher.match(prev_des, des)
#         # matches = [m for m in matches if max_disparity > abs(prev_kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) > min_disparity]
#         matches = [m for m in matches if min_disparity <= m.distance < max_disparity]
#     else:
#         matches = matcher.knnMatch(prev_des, des, k=n_neighbors)
#         best_matches = []
#         for m in matches:
#             valid_matches = [match for match in m if min_disparity <= match.distance < max_disparity]
#             if valid_matches:
#                 best_matches.append(min(valid_matches, key=lambda match: match.distance))
#         matches = best_matches
#     return matches

def match_ratio_test(matcher, prev_des, des, ratio_threshold=0.75):
    matches = matcher.knnMatch(prev_des, des, k=2)
    if len(matches) == 0:
        return []
    return [m for match_set in matches if len(match_set) >= 2 for m, n in [match_set[:2]] if m.distance < ratio_threshold * n.distance]


def filter_matches_by_distance(matches, max_distance=100):
    return [m for m in matches if m.distance < max_distance]


def filter_top_matches(matches, n_top_matches=10):
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:n_top_matches]


def filter_unique_matches(matches):
    matches = sorted(matches, key=lambda x: x.trainIdx)
    return [matches[i] for i in range(len(matches)) if i == 0 or matches[i].trainIdx != matches[i - 1].trainIdx]


def filter_kp_by_std_dist_from_mean(kp, std, mean, n_std=1):
    return [k for k in kp if np.all(np.linalg.norm(np.array(k.pt) - mean) < n_std * std)]


def kp_mean_and_std(keypoints):
    return np.mean(np.array([kp.pt for kp in keypoints]), axis=0), np.std(np.array([kp.pt for kp in keypoints]), axis=0)


def filter_kp_and_des_by_matches(kp, des, matches, is_return_if_no_matches=False):
    """
    takes only the keypoints and descriptors that were matched
    """
    if not is_var_valid(kp):
        return [], np.array([])
    if len(matches) == 0:
        if is_return_if_no_matches:
            return kp, des
        return [], np.array([])
    kp = [kp[m.trainIdx] for m in matches]
    des = np.take(des, [m.trainIdx for m in matches], axis=0)
    return kp, des


def is_var_valid(var):
    # checks if variable is not None and not empty
    if var is None:
        return False
    # if var has attribute __len__ (is iterable) check if it is empty
    if hasattr(var, '__len__'):
        return len(var) > 0
    raise TypeError("var must be iterable or None")


def filter_kp_and_des_by_trainIdx(kp, des, trainIdx, is_return_if_no_matches=False):
    """
    takes only the keypoints and descriptors that were matched
    """
    if not is_var_valid(kp):
        return [], np.array([])
    if not is_var_valid(trainIdx):
        if is_return_if_no_matches:
            return kp, des
        return [], np.array([])
    kp = [kp[i] for i in trainIdx]
    des = np.take(des, trainIdx, axis=0)
    return kp, des


def initialize_orb_trackbars(window_name, callback_func=None):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    def f(x=None):
        return
    if callback_func is not None:
        cv.setMouseCallback(window_name, callback_func)
    cv.createTrackbar('Max Features', window_name, 3000, 5000, f)
    cv.createTrackbar('Scale Factor (x10)', window_name, 15, 40, f)
    cv.createTrackbar('Levels', window_name, 8, 20, f)
    cv.createTrackbar('WTA_K (2 or 4)', window_name, 2, 4, f)
    cv.createTrackbar('edgeThreshold', window_name, 1, 50, f)
    cv.createTrackbar('patchSize', window_name, 31, 100, f)
    cv.createTrackbar('fastThreshold', window_name, 42, 100, f)



def get_orb_params_from_trackbars(window_name):
    nfeatures = cv.getTrackbarPos('Max Features', window_name)
    nfeatures = 10 if nfeatures == 0 else nfeatures
    scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', window_name) / 10.0
    nlevels = cv.getTrackbarPos('Levels', window_name)
    WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', window_name)
    WTA_K = 2 if WTA_K == 2 else 4
    edgeThreshold = cv.getTrackbarPos('edgeThreshold', window_name)
    edgeThreshold = 1 if edgeThreshold == 0 else edgeThreshold
    patchSize = cv.getTrackbarPos('patchSize', window_name)
    patchSize = 2 if patchSize <= 2 else patchSize
    fastThreshold = cv.getTrackbarPos('fastThreshold', window_name)
    fastThreshold = 1 if fastThreshold == 0 else fastThreshold
    return {
        'maxFeatures': nfeatures,
        'scaleFactor': scaleFactor,
        'nLevels': nlevels,
        'WTA_K': WTA_K,
        'edgeThreshold': edgeThreshold,
        'patchSize': patchSize,
        'fastThreshold': fastThreshold
    }


def change_orb_parameters(orb, **orb_params):
    if 'maxFeatures' in orb_params:
        orb.setMaxFeatures(orb_params['maxFeatures'])
    if 'scaleFactor' in orb_params:
        orb.setScaleFactor(orb_params['scaleFactor'])
    if 'nLevels' in orb_params:
        orb.setNLevels(orb_params['nLevels'])
    if 'edgeThreshold' in orb_params:
        orb.setEdgeThreshold(orb_params['edgeThreshold'])
    if 'firstLevel' in orb_params:
        orb.setFirstLevel(orb_params['firstLevel'])
    if 'WTA_K' in orb_params:
        orb.setWTA_K(orb_params['WTA_K'])
    if 'patchSize' in orb_params:
        orb.setPatchSize(orb_params['patchSize'])
    if 'fastThreshold' in orb_params:
        orb.setFastThreshold(orb_params['fastThreshold'])


def get_crop_actual_shape(frame, yx, crop_size):
    y, x = np.array(yx).astype(np.int)
    w, h = crop_size, crop_size
    top_left = np.array([y-h//2, x-w//2])
    # crop the frame until the edges of the frame so the crop is always valid
    top_left = np.maximum(top_left, 0)
    bottom_right = top_left + np.array([h, w])
    bottom_right = np.minimum(bottom_right, np.array(frame.shape[:2]))
    return tuple(bottom_right - top_left)


def crop_frame(frame, yx, crop_size):
    y, x = np.array(yx).astype(np.int)
    w, h = crop_size, crop_size
    top_left = np.array([y-h//2, x-w//2])
    # crop the frame until the edges of the frame so the crop is always valid
    top_left = np.maximum(top_left, 0)
    bottom_right = top_left + np.array([h, w])
    bottom_right = np.minimum(bottom_right, np.array(frame.shape[:2]))
    return (frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]).astype(np.float32), top_left, bottom_right


def crop_frame_multi_channel(frame, yx, crop_size):
    y, x = np.array(yx).astype(np.int)
    w, h = crop_size, crop_size
    top_left = np.array([y-h//2, x-w//2])
    # crop the frame until the edges of the frame so the crop is always valid
    top_left = np.maximum(top_left, 0)
    bottom_right = top_left + np.array([h, w])
    bottom_right = np.minimum(bottom_right, np.array(frame.shape[:2]))
    return (frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]).astype(np.float32), top_left, bottom_right


def correct_kp_coordinates(kp, top_left):
    for k in kp:
        k.pt = (k.pt[0] + top_left[0], k.pt[1] + top_left[1])
    return kp


def calc_gradient_dir(frame, kernel_size):
    # calculating the gradient of the frame using Sobel operator
    # frame: grayscale image
    # returns: gradient direction unit vector as a frame-like image with 2 channels, one for the cos(theta) and one for the sin(theta)
    sobel_x = cv.Sobel(frame, cv.CV_32F, 1, 0, ksize=kernel_size)
    sobel_y = cv.Sobel(frame, cv.CV_32F, 0, 1, ksize=kernel_size)
    gradient = np.stack([sobel_x, sobel_y], axis=2)
    gradient_norm = np.linalg.norm(gradient, axis=2, keepdims=True)
    gradient_norm[gradient_norm == 0] = 1
    gradient /= gradient_norm
    return gradient # gradient direction unit vector


def get_kernel_mean_direction(kernel_grad_dir):
    return normalize_kernel(np.mean(kernel_grad_dir, axis=(0, 1)))


# Calculate the rotation matrix for a given angle in radians
def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


# Rotate a set of vectors
def rotate_vectors(vectors, angle):
    return np.dot(vectors, rotation_matrix(angle).T)


# Given two vectors, find the angle to rotate the first vector to align with the second vector
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    det = np.linalg.det(np.array([v1, v2]))
    angle = np.arctan2(det, dot_product)
    return angle


# Rotate a kernel to align with a given direction
def rotate_kernel_to_align(kernel, target_direction):
    kernel_mean_direction = normalize_kernel(np.mean(kernel, axis=(0, 1)))
    angle = angle_between_vectors(kernel_mean_direction, target_direction)
    rotated_kernel = rotate_vectors(kernel.reshape(-1, 2), -angle)
    return rotated_kernel.reshape(kernel.shape)


# Calculate the mean direction of a window
def get_window_mean_direction(window):
    return normalize_kernel(np.mean(window, axis=(0, 1)))


# Batch rotate all windows to align with a given direction
def batch_rotate_windows(windows, target_direction):
    # Calculate the mean direction for each window
    mean_directions = np.apply_along_axis(get_window_mean_direction, axis=(1, 2), arr=windows)

    # Calculate the angle to rotate each window to align with the target direction
    angles = np.apply_along_axis(angle_between_vectors, axis=1, arr=mean_directions, v2=target_direction)

    # Rotate each window
    rotated_windows = np.apply_along_axis(rotate_vectors, axis=1, arr=windows.reshape(-1, 2), angle=-angles)

    return rotated_windows.reshape(windows.shape)


def draw_osd(img, width, height, radius=10, thickness=2, cross_size=10, outer_radius=5):
    color_gray = (200, 200, 200)
    color_black = (0, 0, 0)
    color_white = (255, 255, 255)
    # draw an X that passes through the center of the screen
    cv.line(img, (0, 0), (width, height), color_gray, 1)
    cv.line(img, (width, 0), (0, height), color_gray, 1)

    # draw a cross at the middle of the screen but without its center. make it out of 4 lines
    cv.line(img, (width // 2 - cross_size, height // 2), (width // 2 - radius - outer_radius, height // 2),
                  color_black, thickness + 2)
    cv.line(img, (width // 2 + cross_size, height // 2), (width // 2 + radius + outer_radius, height // 2),
                  color_black, thickness + 2)
    cv.line(img, (width // 2, height // 2 - cross_size), (width // 2, height // 2 - radius - outer_radius),
                  color_black, thickness + 2)
    cv.line(img, (width // 2, height // 2 + cross_size), (width // 2, height // 2 + radius + outer_radius),
                  color_black, thickness + 2)

    cv.line(img, (width // 2 - cross_size, height // 2), (width // 2 - radius - outer_radius, height // 2),
                  color_white, thickness)
    cv.line(img, (width // 2 + cross_size, height // 2), (width // 2 + radius + outer_radius, height // 2),
                  color_white, thickness)
    cv.line(img, (width // 2, height // 2 - cross_size), (width // 2, height // 2 - radius - outer_radius),
                  color_white, thickness)
    cv.line(img, (width // 2, height // 2 + cross_size), (width // 2, height // 2 + radius + outer_radius),
                  color_white, thickness)
    cv.circle(img, (width // 2, height // 2), radius, color_black, thickness + 1)
    cv.circle(img, (width // 2, height // 2), radius - 1, color_white, thickness)

