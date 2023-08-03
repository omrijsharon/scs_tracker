import cv2 as cv
import utils.screen_capture as sc
import numpy as np

# Initialize the screenshot taker
cap = sc.ScreenCapture(monitor_number=1, tlwh=sc.YOUTUBE_TLWH_SMALL)
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)  # This is for SIFT
# FLANN parameters for ORB
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)

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

cv.createTrackbar('Max Features', 'Matches', 500, 10000, f)
cv.createTrackbar('Scale Factor (x10)', 'Matches', 20, 40, f)
cv.createTrackbar('Levels', 'Matches', 8, 20, f)
cv.createTrackbar('WTA_K (2 or 4)', 'Matches', 2, 4, f)

# Set mouse callback function for window
cv.setMouseCallback('Choose reference image', choose_ref)

while True:
    # Capture screen
    img = cap.capture()

    if start_matching:
        nfeatures = cv.getTrackbarPos('Max Features', 'Matches')
        scaleFactor = cv.getTrackbarPos('Scale Factor (x10)', 'Matches') / 10.0
        nlevels = cv.getTrackbarPos('Levels', 'Matches')
        WTA_K = cv.getTrackbarPos('WTA_K (2 or 4)', 'Matches')
        WTA_K = 2 if WTA_K == 2 else 4

        # Convert images to grayscale
        gray1 = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        detector = cv.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K)

        # Detect keypoints and compute descriptors
        kp1, des1 = detector.detectAndCompute(gray1, None)
        # kp1 = sorted(kp1, key=lambda x: -x.response)[:nfeatures]  # Keep only top N keypoints
        # des1 = detector.compute(gray1, kp1)[1]  # Recompute descriptors for top N keypoints

        kp2, des2 = detector.detectAndCompute(gray2, None)
        # kp2 = sorted(kp2, key=lambda x: -x.response)[:nfeatures]  # Keep only top N keypoints
        # des2 = detector.compute(gray2, kp2)[1]  # Recompute descriptors for top N keypoints

        # kp1, des1 = detector.detectAndCompute(gray1, None)
        # kp2, des2 = detector.detectAndCompute(gray2, None)

        if is_symetric:
            # Match descriptors
            matches12 = bf.match(des1, des2)
            matches21 = bf.match(des1, des2)
            # matches12 = flann.match(des1, des2)
            # matches21 = flann.match(des1, des2)
            # Perform symmetry test
            symmetric_matches = []
            for match_12 in matches12:
                for match_21 in matches21:
                    if match_12.queryIdx == match_21.trainIdx and match_12.trainIdx == match_21.queryIdx:
                        symmetric_matches.append(match_12)
                        break
            matches = symmetric_matches
        else:
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 1.2 * n.distance:
                    good.append([m])
        # Sort matches by distance (smaller is better)
        # matches = sorted(matches, key=lambda x: x.distance)


        # if len(matches) >= 4:
            # After matching descriptors, convert the list of matches to arrays of keypoints.
            # This will allow us to use functions in the cv2 library that take arrays as inputs.
            # points1 and points2 are lists of x, y locations of the keypoints from both images
            # points1 = np.zeros((len(matches), 2), dtype=np.float32)
            # points2 = np.zeros((len(matches), 2), dtype=np.float32)
        #     for i, match in enumerate(matches):
        #         points1[i, :] = kp1[match.queryIdx].pt
        #         points2[i, :] = kp2[match.trainIdx].pt
        #
        #     # Compute homography using RANSAC, this function also provide a mask for outliers
        #     H, mask = cv.findHomography(points1, points2, cv.RANSAC, confidence=0.99, maxIters=1000)
        #
        #     # Use this mask to remove outliers
        #     matchesMask = mask.ravel().tolist()
        #
        #     # Draw matches and the consensus set
        #     # draw_params = dict(singlePointColor=None, matchesMask=matchesMask, flags=2)

        # result = cv.drawMatches(ref_img, kp1, img, kp2, matches, None, **draw_params)
        # result = cv.drawMatches(ref_img, kp1, img, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result = cv.drawMatchesKnn(ref_img, kp1, img, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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