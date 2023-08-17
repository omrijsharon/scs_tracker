import numpy as np
import cv2 as cv
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from time import perf_counter
from utils.helper_functions import json_reader, scale_intrinsic_matrix, create_intrinsic_matrix, match_points, draw_osd
import utils.screen_capture as sc
from utils.pool_helper import slice_image_to_grid_cells, process_cell, calculate_essential_recover_pose

if __name__ == '__main__':
    image_inv_scale = 2 # 2 for 1/2 size, 1 for original size
    n_grid_cells = 4
    n_cells_to_skip_start = 0
    n_cells_to_skip_end = 1
    min_n_matches = 9
    essential_n_processes = 8 # number of processes to use for essential matrix calculation and recover pose to get the average of coordinates
    max_disparity = 27
    marker_size = 10
    p = 0.5
    is_draw_keypoints = True
    matcher_type = 'bf' # 'bf' or 'flann'

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
    # create empty list with size of the grid
    grid_prev_kp = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]
    grid_prev_des = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]
    grid_matches_prev_idx = [[None for _ in range(grid_size[0])] for _ in range(grid_size[1])]

    cv.namedWindow('ORB Detection Test', cv.WINDOW_NORMAL)

    def f(x=None):
        return

    cv.createTrackbar('Max Features', 'ORB Detection Test', 5000, 10000, f)
    cv.createTrackbar('Scale Factor (x10)', 'ORB Detection Test', 20, 40, f)
    cv.createTrackbar('Levels', 'ORB Detection Test', 8, 20, f)
    cv.createTrackbar('WTA_K (2 or 4)', 'ORB Detection Test', 2, 4, f)
    cv.createTrackbar('edgeThreshold', 'ORB Detection Test', 1, 50, f)
    cv.createTrackbar('patchSize', 'ORB Detection Test', 31, 100, f)
    cv.createTrackbar('fastThreshold', 'ORB Detection Test', 50, 100, f)
    cv.createTrackbar('RANSAC subsample_size', 'ORB Detection Test', 250, 1000, f)
    cv.createTrackbar('maxIters', 'ORB Detection Test', 20, 500, f)
    cv.createTrackbar('Min Disparity', 'ORB Detection Test', 7, max_disparity, f)
    cv.createTrackbar('Max Matches per Cell', 'ORB Detection Test', 0, 100, f)
    cv.createTrackbar('Min Matches', 'ORB Detection Test', min_n_matches, 500, f)
    cv.createTrackbar('p', 'ORB Detection Test', int(p * 100), 100, f)
    cv.createTrackbar('draw keypoints?', 'ORB Detection Test', 1, 1, f)


    if matcher_type == 'bf':
        # Create BFMatcher object
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    elif matcher_type == 'flann':
        # FLANN parameters for ORB
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary

        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError('matcher_type must be "bf" or "flann"')

    velocity_dir = None
    pixel_coords, prev_pixel_coords = None, None
    pool = Pool()
    while True:
        img = cap.capture()
        # img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (0, 0), fx=1/image_inv_scale, fy=1/image_inv_scale)

        nfeatures = cv.getTrackbarPos('Max Features', 'ORB Detection Test')
        nfeatures = 10 if nfeatures == 0 else nfeatures
        scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', 'ORB Detection Test') / 10.0
        nlevels = cv.getTrackbarPos('Levels', 'ORB Detection Test')
        WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', 'ORB Detection Test')
        WTA_K = 2 if WTA_K == 2 else 4
        edgeThreshold = cv.getTrackbarPos('edgeThreshold', 'ORB Detection Test')
        edgeThreshold = 31 if edgeThreshold == 0 else edgeThreshold
        patchSize = cv.getTrackbarPos('patchSize', 'ORB Detection Test')
        patchSize = 2 if patchSize <= 2 else patchSize
        fastThreshold = cv.getTrackbarPos('fastThreshold', 'ORB Detection Test')
        fastThreshold = 1 if fastThreshold == 0 else fastThreshold
        subsample_size = cv.getTrackbarPos('RANSAC subsample_size', 'ORB Detection Test')
        subsample_size = 10 if subsample_size <= 10 else subsample_size
        maxIters = cv.getTrackbarPos('maxIters', 'ORB Detection Test')
        maxIters = 1 if maxIters == 0 else maxIters
        min_disparity = cv.getTrackbarPos('Min Disparity', 'ORB Detection Test')
        max_matches_per_cell = cv.getTrackbarPos('Max Matches per Cell', 'ORB Detection Test')
        max_matches_per_cell = -1 if max_matches_per_cell == 0 else max_matches_per_cell
        min_n_matches = cv.getTrackbarPos('Min Matches', 'ORB Detection Test')
        min_n_matches = 10 if min_n_matches < 10 else min_n_matches
        min_n_matches_per_cell = min_n_matches # // (np.prod(np.array(grid_size) - 2))
        p = cv.getTrackbarPos('p', 'ORB Detection Test') / 100.0
        p = 0.01 if p == 0 else p
        is_draw_keypoints = bool(cv.getTrackbarPos('draw keypoints?', 'ORB Detection Test'))

        pts1 = []
        pts2 = []
        t0 = perf_counter()
        # create a dict of the parameters for the ORB detector
        orb_parameters = {'nfeatures': nfeatures // total_cells, 'scaleFactor': scaleFactor, 'nlevels': nlevels, 'WTA_K': WTA_K, 'edgeThreshold': edgeThreshold, 'fastThreshold': fastThreshold}
        t1 = perf_counter() - t0
        # sliced_gray = slice_image_to_grid_cells(gray, cell_width, cell_height, grid_size)

        # prepare arg list for process_cell pool map
        # args_list = [(orb_parameters, gray[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width], (j, i), cell_width, cell_height, grid_prev_kp[j][i], grid_prev_des[j][i], grid_matches_prev_idx[j][i], min_n_matches_per_cell, max_matches_per_cell) for i in range(n_cells_to_skip_start, grid_size[0]-n_cells_to_skip_end) for j in range(n_cells_to_skip_start, grid_size[1]-n_cells_to_skip_end)]
        args_list = [(orb_parameters, gray[j * cell_height + cell_height//2:(j + 1) * cell_height + cell_height//2, i * cell_width + cell_width//2:(i + 1) * cell_width + cell_width//2], (j, i), cell_width, cell_height, grid_prev_kp[j][i], grid_prev_des[j][i], grid_matches_prev_idx[j][i], min_n_matches_per_cell, max_matches_per_cell) for i in range(n_cells_to_skip_start, grid_size[0]-n_cells_to_skip_end) for j in range(n_cells_to_skip_start, grid_size[1]-n_cells_to_skip_end)]

        async_results = [pool.apply_async(process_cell, (args,)) for args in args_list]
        results = [async_result.get() for async_result in async_results]

        # with ThreadPoolExecutor() as executor:
        #     results = list(executor.map(process_cell, args_list))

        # Process the results
        number_of_matches_per_cell = [[result[0], len(result[5])] for result in results]
        for result in results:
            cell_idx, cell_kp, cell_des, cell_grid_matches_prev_idx, cell_pts1, cell_pts2 = result
            j, i = cell_idx
            grid_prev_kp[j][i] = cell_kp
            grid_prev_des[j][i] = cell_des
            grid_matches_prev_idx[j][i] = cell_grid_matches_prev_idx
            pts1.extend(cell_pts1)
            pts2.extend(cell_pts2)

        # for i in range(1, grid_size[0]-1):
        #     for j in range(1, grid_size[1]-1):
        #         cell_idx, grid_prev_kp[j][i], grid_prev_des[j][i], grid_matches_prev_idx[j][i], pts1_array, pts2_array = process_cell(orb_parameters, sliced_gray[j][i], (j, i), cell_width, cell_height, grid_prev_kp[j][i], grid_prev_des[j][i], grid_matches_prev_idx[j][i], min_n_matches_per_cell)
        #         pts1.extend(pts1_array)
        #         pts2.extend(pts2_array)

        t2 = perf_counter() - t0 - t1

        if len(pts1) > 0:
            pts1 = np.vstack(pts1)
            pts2 = np.vstack(pts2)
            # discard points that has a distance smaller than min_disparity or larger than max_disparity using np.linalg.norm(np.array(pt1) - np.array(pt2))
            dist = np.linalg.norm(np.array(pts1) - np.array(pts2), axis=1)
            dist_criteria = np.logical_and(dist > min_disparity, dist < max_disparity)
            pts1 = pts1[dist_criteria]
            pts2 = pts2[dist_criteria]

        if len(pts1) > min_n_matches:
            if len(pts1) > subsample_size:
                # make subsample_size divisible by essential_n_processes without remainder
                subsample_size = subsample_size - (subsample_size % essential_n_processes)
                subsample_idx = np.random.choice(len(pts1), subsample_size, replace=False)
                pts1 = pts1[subsample_idx]
                pts2 = pts2[subsample_idx]
            else:
                # shuffle pts1 and pts2 in the same order
                idx = np.random.permutation(len(pts1))
                pts1 = pts1[idx]
                pts2 = pts2[idx]
            print("len(pts1) per process: ", len(pts1) // essential_n_processes)
            # make arg list use pts1 and pts2 with indices from 0 to subsample_size//essential_n_processes, subsample_size//essential_n_processes to 2*subsample_size//essential_n_processes, etc.
            args_list = [(pts1[i*subsample_size//essential_n_processes:(i+1)*subsample_size//essential_n_processes], pts2[i*subsample_size//essential_n_processes:(i+1)*subsample_size//essential_n_processes], focal, pp, K, width, height, subsample_size, maxIters) for i in range(essential_n_processes)]
            async_results = [pool.apply_async(calculate_essential_recover_pose, (args,)) for args in args_list]
            results = [async_result.get() for async_result in async_results]

            pixel_coords = np.zeros(2)
            n_results = 0
            for result in results:
                result_np = np.array(result)
                pixel_coords += result_np
                n_results += 1 * np.any(result_np != 0)
            p = n_results / essential_n_processes
            print("n_results: ", n_results, "  p = ", p * 100, "%")
            p *= 0.5
            if n_results > 0:
                pixel_coords /= n_results

                t3 = perf_counter() - t0 - t1 - t2
                print(f"keypoint and matches time: {t2 * 1000:.2f} ms",
                      f"essential matrix calculation time: {t3 * 1000:.2f} ms",
                      "total time: {:.2f} ms".format((t1 + t2 + t3) * 1000), " FPS: ", int(1 / (t1 + t2 + t3)))

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
                    # cv.putText(img, str(np.linalg.norm(np.array(pt1) - np.array(pt2)).astype(np.int8)), pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # write a text in the middle of each grid cell of the number of matches in that cell using number_of_matches_per_cell
                for cell_idx, n_matches in number_of_matches_per_cell:
                    j, i = cell_idx
                    if n_matches > 0:
                        # cv.putText(img, str(n_matches), (i * cell_width + cell_width // 2, j * cell_height + cell_height // 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                        # cv.putText(img, str(n_matches), (i * cell_width + cell_width // 2, j * cell_height + cell_height // 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                        cv.putText(img, str(n_matches), (image_inv_scale * i * cell_width + image_inv_scale * cell_width, image_inv_scale * j * cell_height + image_inv_scale * cell_height), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                        cv.putText(img, str(n_matches), (image_inv_scale * i * cell_width + image_inv_scale * cell_width, image_inv_scale * j * cell_height + image_inv_scale * cell_height), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)


        cv.imshow('ORB Detection Test', img)
        if cv.waitKey(10) & 0xFF == 27:
            pool.close()
            pool.join()
            break

    cv.destroyAllWindows()
    cap.close()
    pool.close()
    pool.join()
    # end of the codeth