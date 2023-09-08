import numpy as np
import cv2 as cv

from utils.helper_functions import change_orb_parameters, match_ratio_test, is_var_valid


# slice the image into cells
def slice_image_to_grid_cells(img, cell_width, cell_height, grid_size):
    # do this
    # grid_gray = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]
    # for i in range(1, grid_size[0]-1):
    #     for j in range(1, grid_size[1]-1):
    #         # Compute keypoints and descriptors for each cell
    #         grid_gray[j, i] = img[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width]
    # with numpy slicing and list comprehension:
    return [[img[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width] if 0 < i < grid_size[0]-1 and 0 < j < grid_size[1]-1 else None for i in range(grid_size[0])] for j in range(grid_size[1])]


def process_cell(args):
    orb_parameters, cell_gray, cell_idx, cell_width, cell_height, cell_prev_kp, cell_prev_des, cell_matches_prev_idx, min_n_matches_per_cell, max_matches_per_cell = args
    is_fail_flag = False
    orb = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
    change_orb_parameters(orb, **orb_parameters)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    j, i = cell_idx # cell_idx is a tuple of (row, col)
    pts1_array = np.array([])
    pts2_array = np.array([])
    cell_kp, cell_des = orb.detectAndCompute(cell_gray, None)
    is_any_kp = len(cell_kp) > 0
    is_kp_more_than_min_n_matches = len(cell_kp) >= min_n_matches_per_cell
    if not is_any_kp and not is_kp_more_than_min_n_matches: # there are keypoints but not enough
        for attempt in range(5):
            orb_parameters["fastThreshold"] //= 2
            orb.setFastThreshold(orb_parameters["fastThreshold"]+1)
            cell_kp, cell_des = orb.detectAndCompute(cell_gray, None)
            is_any_kp = len(cell_kp) > 0
            is_kp_more_than_min_n_matches = len(cell_kp) >= min_n_matches_per_cell
            if is_any_kp and is_kp_more_than_min_n_matches:
                break
    if is_any_kp:
        for k in cell_kp:
            k.pt = (k.pt[0] + i * cell_width + cell_width//2, k.pt[1] + j * cell_height + cell_height//2)
            # k.pt = tuple(map(sum, zip(k.pt, (i * cell_width, j * cell_height))))
        if cell_prev_kp is not None:  # not first frame
            cell_prev_kp = convert_tuple_to_keypoints(cell_prev_kp)
            # if len(cell_prev_kp) > 1 and len(cell_prev_des) > 1 and is_kp_more_than_min_n_matches:
            if len(cell_prev_kp) > 1 and len(cell_prev_des) > 1:
                if cell_matches_prev_idx is None:  # first pair of frames
                    prev_des = cell_prev_des
                else:
                    # check if all the indices in grid_matches_prev_idx[j][i] are valid
                    valid_matches = ([0 <= idx < len(cell_prev_des) for idx in cell_matches_prev_idx])
                    if len(cell_matches_prev_idx) <= min_n_matches_per_cell or any(valid_matches):
                        prev_des = cell_prev_des
                    else:
                        cell_matches_prev_idx = np.array(cell_matches_prev_idx)[valid_matches]
                        prev_des = np.take(cell_prev_des, cell_matches_prev_idx, axis=0)
                matches = match_ratio_test(matcher, prev_des, cell_des)
                # handle if there are no matches or only one match in knn
                if len(matches) == 0:
                    is_fail_flag = True
                else:
                    # matches = sorted(matches, key=lambda x: x.distance)[:max_matches_per_cell]
                    cell_matches_prev_idx = [m.trainIdx for m in matches]
                    if len(cell_matches_prev_idx) == 0:
                        is_fail_flag = True
                    else:
                        pts1_array = np.float32([cell_prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                        pts2_array = np.float32([cell_kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                        # pass only the kp and des that were matched:
                        # cell_kp = [cell_kp[m.trainIdx] for m in matches]
                        # cell_des = np.take(cell_des, cell_matches_prev_idx, axis=0)


        if is_fail_flag:
            cell_prev_kp = None
            cell_prev_des = None
            cell_matches_prev_idx = None
        else:
            # assign cell_kp to grid_prev_kp
            cell_prev_kp = convert_keypoints_to_tuple(cell_kp)
            cell_prev_des = cell_des
    else:
        cell_prev_kp = None
        cell_prev_des = None
        cell_matches_prev_idx = None
    return cell_idx, cell_prev_kp, cell_prev_des, cell_matches_prev_idx, pts1_array, pts2_array


def process_cell_v2(args):
    cell_idx, cell_kp, cell_des, cell_prev_kp, cell_prev_des, cell_matches_prev_idx, min_n_matches_per_cell, max_matches_per_cell = args
    if cell_kp is not None:
        cell_kp = convert_tuple_to_keypoints(cell_kp)
    is_fail_flag = False
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    pts1_array = np.array([])
    pts2_array = np.array([])
    is_any_kp = len(cell_kp) > 0
    is_kp_more_than_min_n_matches = len(cell_kp) >= min_n_matches_per_cell
    if is_any_kp:
        if cell_prev_kp is not None:  # not first frame
            cell_prev_kp = convert_tuple_to_keypoints(cell_prev_kp)
            # if len(cell_prev_kp) > 1 and len(cell_prev_des) > 1 and is_kp_more_than_min_n_matches:
            if len(cell_prev_kp) > 1 and len(cell_prev_des) > 1:
                if cell_matches_prev_idx is None:  # first pair of frames
                    prev_des = cell_prev_des
                else:
                    # check if all the indices in grid_matches_prev_idx[j][i] are valid
                    valid_matches = ([0 <= idx < len(cell_prev_des) for idx in cell_matches_prev_idx])
                    if len(cell_matches_prev_idx) <= min_n_matches_per_cell or any(valid_matches):
                        prev_des = cell_prev_des
                    else:
                        cell_matches_prev_idx = np.array(cell_matches_prev_idx)[valid_matches]
                        prev_des = np.take(cell_prev_des, cell_matches_prev_idx, axis=0)
                matches = match_ratio_test(matcher, prev_des, cell_des, ratio_threshold=0.6)
                # handle if there are no matches or only one match in knn
                if len(matches) == 0:
                    is_fail_flag = True
                else:
                    matches = sorted(matches, key=lambda x: x.distance)[:max_matches_per_cell]
                    # cell_matches_prev_idx = [m.trainIdx for m in matches]
                    if is_var_valid(cell_matches_prev_idx):
                        is_fail_flag = True
                    else:
                        pts1_array = np.float32([cell_prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                        pts2_array = np.float32([cell_kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                        # pass only the kp and des that were matched:
                        # cell_kp = [cell_kp[m.trainIdx] for m in matches]
                        # cell_des = np.take(cell_des, cell_matches_prev_idx, axis=0)

        if is_fail_flag:
            cell_prev_kp = None
            cell_prev_des = None
            cell_matches_prev_idx = None
        else:
            # assign cell_kp to grid_prev_kp
            cell_prev_kp = convert_keypoints_to_tuple(cell_kp)
            cell_prev_des = cell_des
    else:
        cell_prev_kp = None
        cell_prev_des = None
        cell_matches_prev_idx = None
    return cell_idx, cell_prev_kp, cell_prev_des, cell_matches_prev_idx, pts1_array, pts2_array


def convert_keypoints_to_tuple(kp):
    return tuple({'angle': k.angle, 'class_id': k.class_id, 'octave': k.octave, 'x': k.pt[0], 'y': k.pt[1], 'response': k.response, 'size': k.size} for k in kp)

def convert_tuple_to_keypoints(kp):
    return [cv.KeyPoint(**k) for k in kp]


def calculate_essential_recover_pose(args):
    pts1, pts2, focal, pp, K, width, height, subsample_size, maxIters = args
    is_use_reprojection_error = False
    error_threshold = 3
    pixel_coords = (0, 0)
    kp_depth = None
    if len(pts1) > 8:
        E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, method=cv.FM_RANSAC, prob=0.999999, threshold=1,
                                      maxIters=maxIters)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        if len(pts1) > 8:
            if not is_use_reprojection_error:
                _, R, t, _ = cv.recoverPose(E, pts1, pts2)
            else:

                _, R, t, _ = cv.recoverPose(E, pts1, pts2)

                errors = calculate_reprojection_error(pts1, pts2, R, t, K)
                # mask pts1 and pts2 by errors with error threshold
                pts1 = pts1[errors < error_threshold]
                pts2 = pts2[errors < error_threshold]
                errors = errors[errors < error_threshold]
                if len(pts1) > 8:
                    E, mask = cv.findEssentialMat(pts1, pts2, focal, pp, method=cv.FM_RANSAC, prob=0.999999, threshold=1,
                                                  maxIters=maxIters)
                    pts1 = pts1[mask.ravel() == 1]
                    pts2 = pts2[mask.ravel() == 1]
                    print("len(pts2) per process:", len(pts2))
                    _, R, t, _ = cv.recoverPose(E, pts1, pts2)

            velocity_dir = R.dot(t.reshape(3, 1))
            velocity_dir = velocity_dir / np.linalg.norm(velocity_dir)
            pixel_coords_hom = np.dot(K, velocity_dir)
            pixel_coords = (pixel_coords_hom[0:2] / pixel_coords_hom[2]).astype(int).flatten()

            if len(pixel_coords) == 4:
                pixel_coords = [pixel_coords[0], pixel_coords[2]]

            pixel_coords[0] = max(0, min(pixel_coords[0], width))
            pixel_coords[1] = max(0, min(pixel_coords[1], height))
            # Compute depth
            # extrinsic = np.hstack([R, t])
            # P1 = np.dot(K, np.eye(3, 4))
            # P2 = np.dot(K, extrinsic)
            # homogeneous_3D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
            # dehomo_3D = (homogeneous_3D / homogeneous_3D[3]).T
            # depths = dehomo_3D[:, 2]
            # # get an array in a format of [pixel_y, pixel_y, depth]
            # kp_depth = np.hstack([pts2, depths.reshape(-1, 1)])
    # return pixel_coords, kp_depth
    return pixel_coords, pts1, pts2


def calculate_reprojection_error(pts1, pts2, R2, t2, K):
    """
    Calculate reprojection errors after triangulating 3D points.

    pts1, pts2: Nx2 arrays of 2D points in the first and second images
    R1, t1: Rotation and translation of the first camera
    R2, t2: Rotation and translation of the second camera
    K: Intrinsic camera matrix
    """
    # Create projection matrices
    P1 = np.dot(K, np.eye(3, 4))
    P2 = np.dot(K, np.hstack((R2, t2)))

    # Triangulate 3D points
    # since R1 is always the identity matrix and t1 is always zero vector, P1 is always K
    pts4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3, :] / pts4D[3, :]
    pts3D = pts3D.T

    # Reproject to 2D in the second image
    reprojected_pts = cv.projectPoints(pts3D, R2, t2, K, None)[0].reshape(-1, 2)

    # Calculate errors
    errors = np.linalg.norm(pts2 - reprojected_pts, axis=1)

    return errors