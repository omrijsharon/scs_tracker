import cv2 as cv
import utils.screen_capture as sc
import numpy as np

from utils.helper_functions import match_ratio_test

# Initialize the screenshot taker
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
detector = cv.SIFT_create(nfeatures=100, contrastThreshold=0.04, sigma=1.6, nOctaveLayers=3, edgeThreshold=10)
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  # This is for SIFT
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

# This flag and the ref_img will control when we start matching
start_matching = False
ref_img = None

is_symetric = False


# Mouse callback function
def choose_ref(event, x, y, flags, param):
    global start_matching, ref_img
    if event == cv.EVENT_LBUTTONDOWN:
        start_matching = True
        ref_img = cap.capture()
        # destroy the window named 'Choose reference image'
        cv.destroyWindow('Choose reference image')

def f(x=None):
    return

# Named window
cv.namedWindow('Choose reference image')
cv.namedWindow('Matches', cv.WINDOW_NORMAL)

cv.createTrackbar('Max Features', 'Matches', 500, 5000, f)
cv.createTrackbar('ratio threshold', 'Matches', 90, 100, f)

# Set mouse callback function for window
cv.setMouseCallback('Choose reference image', choose_ref)

while True:
    # Capture screen
    img = cap.capture()

    if start_matching:
        nfeatures = cv.getTrackbarPos('Max Features', 'Matches')
        # read ratio threshold trackbar
        ratio_threshold = cv.getTrackbarPos('ratio threshold', 'Matches') / 100.0
        ratio_threshold = 0.01 if ratio_threshold == 0 else ratio_threshold
        ratio_threshold = 0.99 if ratio_threshold == 1 else ratio_threshold
        # Convert images to grayscale
        gray1 = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # detector = cv.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K)

        # Detect keypoints and compute descriptors
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp1 = sorted(kp1, key=lambda x: -x.response)[:nfeatures]  # Keep only top N keypoints
        des1 = detector.compute(gray1, kp1)[1]  # Recompute descriptors for top N keypoints

        kp2, des2 = detector.detectAndCompute(gray2, None)
        kp2 = sorted(kp2, key=lambda x: -x.response)[:nfeatures]  # Keep only top N keypoints
        des2 = detector.compute(gray2, kp2)[1]  # Recompute descriptors for top N keypoints

        # kp1, des1 = detector.detectAndCompute(gray1, None)
        # kp2, des2 = detector.detectAndCompute(gray2, None)

        # Match descriptors
        if not is_symetric:
            # matches = match_ratio_test(bf, des2, des1, ratio_threshold=ratio_threshold)
            # matches = bf.match(des2, des1)
            matches = flann.match(des2, des1)
        else:
            matches12 = flann.knnMatch(des1,des2,k=2)
            matches21 = flann.knnMatch(des2,des1,k=2)
            # Perform symmetry test
            symmetric_matches = []
            for match_12 in matches12:
                for match_21 in matches21:
                    if match_12.queryIdx == match_21.trainIdx and match_12.trainIdx == match_21.queryIdx:
                        symmetric_matches.append(match_12)
                        break
            matches = symmetric_matches
        # Sort matches by distance (smaller is better)
        # matches = sorted(matches, key=lambda x: x.distance)

        # After matching descriptors, convert the list of matches to arrays of keypoints.
        # This will allow us to use functions in the cv2 library that take arrays as inputs.
        # points1 and points2 are lists of x, y locations of the keypoints from both images
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        if len(matches) >= 4:
            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.trainIdx].pt
                points2[i, :] = kp2[match.queryIdx].pt

            # Compute homography using RANSAC, this function also provide a mask for outliers
            H, mask = cv.findHomography(points2, points1, cv.RANSAC, confidence=0.99, maxIters=100)
            # aligned_img2 = cv.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))
            # img = cv.addWeighted(img, 0.5, aligned_img2, 0.5, 0)
            # Use this mask to remove outliers
            matchesMask = mask.ravel().tolist()

            # Draw matches and the consensus set
            draw_params = dict(singlePointColor=None, matchesMask=matchesMask, flags=0)

            result = cv.drawMatches(img, kp2, ref_img, kp1, matches, None, **draw_params)
            # result = cv.drawMatches(ref_img, kp1, img, kp2, matches, None)
            # img = cv.drawKeypoints(img, kp1, None, color=(0, 255, 0))
            # Show the matches
            cv.imshow('Matches', result)
            # cv.imshow('Matches', img)
    else:
        # Show the image and wait for click
        cv.imshow('Choose reference image', img)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()