import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
Image = Union[Mat, np.ndarray]


def bgr_to_yiq(bgr_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Transformation matrix for RGB to YIQ
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [0.596, -0.274, -0.322],
                                 [0.211, -0.523, 0.312]])

    # Apply the transformation matrix to each pixel
    yiq_image = np.dot(rgb_image, transform_matrix.T)
    yiq_image = np.clip(yiq_image, 0, 255)  # Ensure the values are within byte range

    return yiq_image




def subtract_images(image1: Image, image2: Image) -> Image:
    image2_with_neg = image1.astype(np.int32)
    image1_with_neg = image2.astype(np.int32)
    return np.abs(image2_with_neg-image1_with_neg).astype(np.uint8)


def get_edge_image(balls_image: Image, original_image: Image) -> Union[List[Tuple[int]], Image]:
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    subtracted_image = subtract_images(balls_image, original_image)
    rgb, hsv, bilateral_color, gray = images_formats(subtracted_image)
    cv2.imshow("bilateral_color", bilateral_color)
    edges = cv2.Canny((gray), 0, 50)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)
    return edges



def find_circle(image: Image):
    pass

def find_cue():
    pass


def find_balls(edge_image: Image):
    no_lines = find_lines_and_remove(edge_image)


def images_formats(image: Image) -> Tuple[Image, Image, Image, Image]:
    """
    param1: image
    return: Tuple[Image, Image, Image, Image]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bilateral_color = cv2.bilateralFilter(rgb, 9, 100, 20)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)
    return rgb, hsv, bilateral_color, gray






def find_lines_and_remove(edges) -> Image:
    """
    :param image: Image
    :return: List of tuples containing the coordinates of the lines
    """
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # creating a blank to draw lines on
    line_image = np.copy(edges) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    #reduce the only line image from the edges
    without_lines = cv2.subtract(edges, line_image)
    return without_lines



board_image = cv2.imread(r"photos_1\WIN_20240214_22_28_47_Pro.jpg")
balls_image = cv2.imread(r"photos_1\WIN_20240214_22_29_42_Pro.jpg")
edge_image = get_edge_image(balls_image, board_image)
find_balls(edge_image)