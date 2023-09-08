import numpy as np
import cv2 as cv
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

#@TODO: change the kp tracking from matching kp to scs tracker

from time import perf_counter
from utils.helper_functions import json_reader, scale_intrinsic_matrix, create_intrinsic_matrix, draw_osd, \
    initialize_orb_trackbars, get_orb_params_from_trackbars, filter_kp_and_des_by_trainIdx, is_var_valid, \
    change_orb_parameters, calc_scs, crop_frame, get_crop_actual_shape, scs_matcher, scs_matcher_pool
import utils.screen_capture as sc
from utils.pool_helper import slice_image_to_grid_cells, process_cell, calculate_essential_recover_pose, \
    convert_keypoints_to_tuple, process_cell_v2


def get_kp(orb, gray, width, height, max_n_kp=200, kp_max_dist=21):
    kp, des = orb.detectAndCompute(gray, None)
    # get the 17 kp that are closest to the center of the image
    kp = sorted(kp, key=lambda x: np.linalg.norm(np.array(x.pt) - np.array([width / 2, height / 2])))
    kp = kp[:max_n_kp]
    # leave only kp that are at least 21 pixels apart from each other, else delete one of them
    # for i in range(len(kp)):
    #     for j in range(i + 1, len(kp)):
    #         if np.linalg.norm(np.array(kp[i].pt) - np.array(kp[j].pt)) < kp_max_dist:
    #             kp.pop(j)
                # break
    return kp


if __name__ == '__main__':
    image_inv_scale = 2 # 2 for 1/2 size, 1 for original size
    n_grid_cells = 4
    n_cells_to_skip_start = 0
    n_cells_to_skip_end = 1
    min_n_matches = 9
    essential_n_processes = 4 # number of processes to use for essential matrix calculation and recover pose to get the average of coordinates
    max_disparity = 27
    marker_size = 10
    p = 0.5
    is_draw_keypoints = True
    crop_list = []
    kernel_list = []
    next_frame_kernel_list = []
    top_left_list = []
    yx_list = []
    crop_size = 91
    kernel_size = 21
    max_n_kp = 63
    kp_max_dist = 51
    cross_size = 20
    color_white = (255, 255, 255)
    color_gray = (200, 200, 200)
    color_black = (0, 0, 0)
    thickness = 2
    radius = 2
    outer_radius = 5

    cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)

    hfov = 120
    cell_size = (cap.monitor["width"] / n_grid_cells, cap.monitor["height"] / n_grid_cells)
    img = cap.capture()
    # resize image to half its size
    img = cv.resize(img, (0, 0), fx=1/image_inv_scale, fy=1/image_inv_scale)
    v_fov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) * img.shape[0] / img.shape[1]))
    K = create_intrinsic_matrix(*img.shape[:2][::-1], hfov, vfov=v_fov)
    focal = K[0, 0]
    pp = (K[0, 2], K[1, 2])

    # define the size of the grid
    grid_size = np.array((n_grid_cells, n_grid_cells))
    total_cells = np.prod(grid_size-2) # total number of cells in the grid without the outermost cells
    height, width = img.shape[:2]
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    depth_image = np.zeros(img.shape[:2])
    window_name = 'SCS Motion Estimator'
    initialize_orb_trackbars(window_name, callback_func=None)

    def f(x=None):
        return

    cv.createTrackbar('RANSAC subsample_size', window_name, 1000, 5000, f)
    cv.createTrackbar('maxIters', window_name, 10, 500, f)
    cv.createTrackbar('Min Disparity', window_name, 7, max_disparity, f)
    cv.createTrackbar('Max Matches per Cell', window_name, 0, 100, f)
    cv.createTrackbar('Min Matches', window_name, min_n_matches, 500, f)
    cv.createTrackbar('p', window_name, int(p * 100), 100, f)
    cv.createTrackbar('draw keypoints?', window_name, 1, 1, f)

    velocity_dir = None
    pixel_coords, prev_pixel_coords = None, None
    pool = Pool()

    orb = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE)
    img = cap.capture()
    # img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (0, 0), fx=1 / image_inv_scale, fy=1 / image_inv_scale)
    kp = get_kp(orb, gray, width, height, max_n_kp=max_n_kp, kp_max_dist=kp_max_dist)
    for kp_idx in range(len(kp)):
        yx = kp[kp_idx].pt[::-1]
        # check if kernel is really size of kernel size
        kernel_shape = get_crop_actual_shape(gray, yx, kernel_size)
        if kernel_shape[0] != kernel_size or kernel_shape[1] != kernel_size:
            continue
        yx_list.append(yx)
        kernel, _, _ = crop_frame(gray, yx, kernel_size)
        kernel_list.append(kernel)

    while True:
        img = cap.capture()
        # img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (0, 0), fx=1/image_inv_scale, fy=1/image_inv_scale)
        crop_list.clear()
        top_left_list.clear()
        for kp_idx in range(len(kp)):
            yx = kp[kp_idx].pt[::-1]
            # check if kernel is really size of kernel size
            kernel_shape = get_crop_actual_shape(gray, yx, kernel_size)
            if kernel_shape[0] != kernel_size or kernel_shape[1] != kernel_size:
                continue
            cropped_frame, top_left, bottom_right = crop_frame(gray, yx, crop_size)
            crop_list.append(cropped_frame)
            top_left_list.append(top_left)

        # create a dict of the parameters for the ORB detector
        orb_parameters = get_orb_params_from_trackbars(window_name)
        # delete patchSize from orb_parameters
        del orb_parameters['patchSize']
        change_orb_parameters(orb, **orb_parameters)

        subsample_size = cv.getTrackbarPos('RANSAC subsample_size', window_name)
        subsample_size = 10 if subsample_size <= 10 else subsample_size
        maxIters = cv.getTrackbarPos('maxIters', window_name)
        maxIters = 1 if maxIters == 0 else maxIters
        min_disparity = cv.getTrackbarPos('Min Disparity', window_name)
        max_matches_per_cell = cv.getTrackbarPos('Max Matches per Cell', window_name)
        max_matches_per_cell = -1 if max_matches_per_cell == 0 else max_matches_per_cell
        min_n_matches = cv.getTrackbarPos('Min Matches', window_name)
        min_n_matches = 10 if min_n_matches < 10 else min_n_matches
        min_n_matches_per_cell = min_n_matches # // (np.prod(np.array(grid_size) - 2))
        p = cv.getTrackbarPos('p', window_name) / 100.0
        p = 0.01 if p == 0 else p
        is_draw_keypoints = bool(cv.getTrackbarPos('draw keypoints?', window_name))

        pts1 = []
        pts2 = []
        # sliced_gray = slice_image_to_grid_cells(gray, cell_width, cell_height, grid_size)
        t0 = perf_counter()

        # create arg_list for scs_matcher(cropped_frame, kernel, top_left, p=3, q=1e-6)
        args_list = [(crop_list[i], kernel_list[i], top_left_list[i], 3, 1e-6) for i in range(len(crop_list))]

        async_results = [pool.apply_async(scs_matcher_pool, (args,)) for args in args_list]
        new_yx_list = np.array([async_result.get() for async_result in async_results])
        t1 = perf_counter() - t0

        # with ThreadPoolExecutor() as executor:
        #     results = list(executor.map(process_cell, args_list))

        if len(yx_list) > 0:
            pts1 = np.vstack(yx_list)[:, ::-1]
            pts2 = np.vstack(new_yx_list)[:, ::-1]

            # discard points that has a distance smaller than min_disparity or larger than max_disparity using np.linalg.norm(np.array(pt1) - np.array(pt2))
            dist = np.linalg.norm(np.array(pts1) - np.array(pts2), axis=1)
            dist_criteria = np.logical_and(dist > min_disparity, dist < max_disparity)
            pts1 = pts1[dist_criteria]
            pts2 = pts2[dist_criteria]

        print("len(pts1):", len(pts1), "len(pts2):", len(pts2))
        if len(pts1) > min_n_matches:
            args = pts1, pts2, focal, pp, K, width, height, subsample_size, maxIters
            pixel_coords, pts1, pts2 = calculate_essential_recover_pose(args)
            pixel_coords = np.array(pixel_coords)
            t2 = perf_counter() - t0 - t1
            if len(pixel_coords) != 0:
                # Draw the cross at the projected point
                draw_osd(img, width * image_inv_scale, height * image_inv_scale, radius=radius, thickness=thickness, cross_size=cross_size, outer_radius=outer_radius)
                # draw a point in the middle of the screen
                img = cv.circle(img, (image_inv_scale * width // 2, image_inv_scale * height // 2), radius, color_black, thickness + 1)
                img = cv.circle(img, (image_inv_scale * width // 2, image_inv_scale * height // 2), radius - 1, color_white, thickness)

                # draw a circle at the projected point
                if prev_pixel_coords is None:  # first frame
                    prev_pixel_coords = np.array(pixel_coords)
                else:
                    pixel_coords = (p * np.array(pixel_coords) + (1 - p) * prev_pixel_coords).astype(int)
                    prev_pixel_coords = pixel_coords
                pixel_coords = pixel_coords.astype(int)
                img = cv.circle(img, (image_inv_scale * pixel_coords[0], image_inv_scale * pixel_coords[1]), marker_size, color_black, thickness + 2)
                img = cv.circle(img, (image_inv_scale * pixel_coords[0], image_inv_scale * pixel_coords[1]), marker_size, color_white, thickness)

        if len(pts2)>0:
            fps = int(1 / (perf_counter() - t0))
            # add text to image with len(matches)
            cv.putText(img, str(len(pts2)), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(img, str(fps) + "FPS", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 4, cv.LINE_AA)

            cv.putText(img, str(len(pts2)), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            # add text of fps which is 1/(perf_counter() - t0)
            cv.putText(img, str(fps) + "FPS", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        if is_draw_keypoints:
            if len(pts2) > min_n_matches:
                # draw a line between each keypoint and its maching keypoint from the previous frame
                # Loop over the matches and draw lines between matching points
                for pt1, pt2 in zip(pts1, pts2):
                    pt1 = tuple(np.round(image_inv_scale * pt1).astype(int))
                    pt2 = tuple(np.round(image_inv_scale * pt2).astype(int))
                    # cv.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.3)
                    cv.arrowedLine(img, pt2, pt1, (0, 255, 0), 2, tipLength=0.3)
                    # cv.putText(img, str(np.linalg.norm(np.array(pt1) - np.array(pt2)).astype(np.int8)), pt2, cv.FONT_HERSHEY_SIMPLE

        # create a crop and a kernel around each keypoint with sizes crop_size and kernel_size respectively
        kp = get_kp(orb, gray, width, height, max_n_kp=max_n_kp, kp_max_dist=kp_max_dist)
        yx_list.clear()
        kernel_list.clear()
        for kp_idx in range(len(kp)):
            yx = kp[kp_idx].pt[::-1]
            # check if kernel is really size of kernel size
            kernel_shape = get_crop_actual_shape(gray, yx, kernel_size)
            if kernel_shape[0] != kernel_size or kernel_shape[1] != kernel_size:
                continue
            yx_list.append(yx)
            kernel, _, _ = crop_frame(gray, yx, kernel_size)
            kernel_list.append(kernel)
        if len(pts1) > min_n_matches:
            t3 = perf_counter() - t0 - t1 - t2
            print(f"   SCS matching time: {t1 * 1000:.2f} ms",
                  f"   essential matrix + recover pose time calculation time: {t2 * 1000:.2f} ms",
                  f"   keypoints detection time: {t3 * 1000:.2f} ms",
                  "   total time: {:.2f} ms".format((t1 + t2 + t3) * 1000), " FPS: ", int(1 / (t1 + t2 + t3)))

        cv.imshow(window_name, img)
        if cv.waitKey(1) & 0xFF == 27:
            pool.close()
            pool.join()
            break

    cv.destroyAllWindows()
    cap.close()
    pool.close()
    pool.join()
    # end of the codeth