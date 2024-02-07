from main import Ball
import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
Image = Union[Mat, np.ndarray]



def find_balls(image: Image) -> List[Tuple[int]]:
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    cut_and_resized_image = cut_and_resize(image)
    cut_and_resized_image, rgb, hsv, bilateral_color, gray = images_formats(cut_and_resized_image)
    edges = cv2.Canny(bilateral_color, 200, 100)
    contours = line_detected(edges)
    drawn_centers, original_image_with_mark = find_ellipses(bilateral_color, contours, edges, image.copy())
    cv2.imshow("original_image_with_mark", original_image_with_mark)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)

def cut_and_resize(image: Image) -> Image:
    # resize the image to 75%
    scale_percent = 75
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # change the image
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    image_without_frame = image.copy()
    thickness = 13
    # Black out the top, bottom, left, and right parts
    image_without_frame[:thickness, :] = 0  # Top
    image_without_frame[-thickness:, :] = 0  # Bottom
    image_without_frame[:, :thickness] = 0  # Left
    image_without_frame[:, -thickness:] = 0  # Right
    return image_without_frame

def images_formats(image: Image) -> Tuple[Image]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bilateral_color = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)
    return image, rgb, hsv, bilateral_color, gray

def dilation_and_erosion(image: Image) -> Image:
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion


def line_detected(edges) -> Image:
    # Parameters for line detection
    rho = 1  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians
    threshold = 50  # Minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # Minimum number of pixels making up a line
    max_line_gap = 10  # Maximum gap in pixels between connectable line segments

    # Detect lines in the image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # Erase the detected lines
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 0), 3)  # Drawing a black line over the detected line
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_ellipses(image: Image, contours, edges, original_image) -> Union[Image, List[Tuple[int]]]:
    min_major_axis = 5 # Minimum length of the major axis
    max_major_axis = 1000  # Maximum length of the major axis
    min_minor_axis = 20  # Minimum length of the minor axis
    max_minor_axis = 50  # Maximum length of the minor axis

    min_distance = 20  # Example value, adjust as needed

    # Store the centers that have been drawn
    drawn_centers = []

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse

            if min_major_axis <= MA <= max_major_axis and min_minor_axis <= ma <= max_minor_axis:
                center = (int(x), int(y))

                # Check the distance to previously drawn centers
                too_close = any(np.linalg.norm(np.array(center) - np.array(c)) < min_distance for c in drawn_centers)

                if not too_close:
                    drawn_centers.append(center)
                    # Draw the red point
                    cv2.circle(original_image, center, 2, (0, 0, 255), -1)

    return drawn_centers, original_image

image = cv2.imread("images/board2.jpg")
find_balls(image)