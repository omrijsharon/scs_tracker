import cv2 as cv
import numpy as np

# create an ArUco dictionary with 5x5 bits and 250 markers
arucodicts = {"DICT_4X4_50": cv.aruco.DICT_4X4_50,
              "DICT_4X4_100": cv.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv.aruco.DICT_4X4_250,
              "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv.aruco.DICT_5X5_50,
              "DICT_5X5_100": cv.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv.aruco.DICT_5X5_250,
              "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv.aruco.DICT_6X6_50,
              "DICT_6X6_100": cv.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv.aruco.DICT_6X6_250,
              "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv.aruco.DICT_7X7_50,
              "DICT_7X7_100": cv.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv.aruco.DICT_7X7_250,
              "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
              }

aruco_type = "DICT_5X5_250"
aruco_dict = cv.aruco.getPredefinedDictionary(arucodicts[aruco_type])
id = 249 # This is the identifier of the bookmark, you can change it to whatever you need
tag_size = 1000 # Define the size of the final image
tag = cv.aruco.generateImageMarker(aruco_dict, id, tag_size)
# show aruco tag
cv.imshow("tag", tag)
cv.waitKey(0)
cv.destroyAllWindows()