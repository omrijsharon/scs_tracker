import cv2 as cv
import numpy as np


is_dots = False

# Function to add center dots to an ArUco marker image
def add_center_dot(marker_img, dot_radius=10, color=0):
    """
    Adds a center dot to the ArUco marker image.

    Parameters:
        marker_img: ndarray
            The ArUco marker image.
        dot_radius: int, optional
            The radius of the center dot.
        color: int, optional
            The color of the center dot (0 for black, 255 for white).

    Returns:
        ndarray: The ArUco marker image with a center dot.
    """
    center_x, center_y = marker_img.shape[1] // 2, marker_img.shape[0] // 2
    cv.circle(marker_img, (center_x, center_y), dot_radius, color, -1)
    return marker_img


# Dictionary of ArUco markers
arucodicts = {
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    # Add other dictionaries as needed
}

# Choose ArUco type and get dictionary
aruco_type = "DICT_5X5_250"
aruco_dict = cv.aruco.getPredefinedDictionary(arucodicts[aruco_type])

# Generate four different ArUco markers
marker_ids = [0, 1, 2, 3, 4]
tag_size = 700  # Change this based on your monitor size
tag_canvas_size = 3840  # 4K width
corner_distance = 100

# Create a white canvas for 4K resolution
canvas = 255 * np.ones((2160, tag_canvas_size), dtype=np.uint8)

for idx, mid in enumerate(marker_ids):
    # Generate individual ArUco marker
    tag = cv.aruco.generateImageMarker(aruco_dict, mid, tag_size)
    # Add a black center dot with a white dot on top
    if is_dots:
        tag = add_center_dot(tag, dot_radius=30, color=0)
        tag = add_center_dot(tag, dot_radius=10, color=255)
    # Get position to place the marker on canvas
    if idx == 4:  # Center marker
        x_offset = 3840 // 2 - tag_size // 2
        y_offset = 2160 // 2 - tag_size // 2
    else:
        x_offset = corner_distance if idx % 2 == 0 else 3840 - tag_size - corner_distance
        y_offset = corner_distance if idx < 2 else 2160 - tag_size - corner_distance
    if idx == 4:
        # Place marker on canvas
        canvas[y_offset:y_offset + tag_size, x_offset:x_offset + tag_size] = tag

if is_dots:
    cv.imwrite(r"C:\Users\omri_\OneDrive\Documents\4K_ArUco_Markers_w_dots.png", canvas)
else:
    cv.imwrite(r"C:\Users\omri_\OneDrive\Documents\4K_ArUco_Markers.png", canvas)
# Display canvas with markers
# cv.imshow("4K ArUco Markers", canvas)
# cv.waitKey(0)
# cv.destroyAllWindows()