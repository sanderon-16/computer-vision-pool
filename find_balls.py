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
    yiq = bgr_to_yiq(bilateral_color)

    edges = cv2.Canny(gray, 200, 100)

    cv2.imshow("edges", edges)
    cv2.imshow("yiq", yiq)
    cv2.imshow("original", image)
    cv2.waitKey(0)


    without_lines = find_lines_and_remove(edges)
    #contours = line_detected(edges)
    #drawn_centers, original_image_with_mark = find_ellipses(bilateral_color, contours, edges, image.copy())


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



def find_lines_and_remove(edges) -> List[Tuple[int]]:
    """
    :param image: Image
    :return: List of tuples containing the coordinates of the lines
    """
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
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

def yiq_to_gray(yiq_image):
    # Extract the Y channel (luminance)
    y_channel = yiq_image[:, :, 0]

    # Convert the Y channel to grayscale
    gray_image = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
    return gray_image

image = cv2.imread("images\photo_from_camera.jpg")
find_balls(image)