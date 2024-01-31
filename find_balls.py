from main import Ball
import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
Image = Union[Mat, np.ndarray]



def find_balls(image: Image) -> List[Ball]:
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bilateral_color = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.imshow('bilateral_color', bilateral_color)
    cv2.waitKey(0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)

    # dilation and erosion
    kernel = np.ones((3,3), np.uint8)
    dilate_gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((3,3), np.uint8)
    erode_gray = cv2.erode(dilate_gray, kernel, iterations=1)

    #canny and find contours
    edges = cv2.Canny(bilateral_color, 200, 100)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_major_axis = 2 # Minimum length of the major axis
    max_major_axis = 40  # Maximum length of the major axis
    min_minor_axis = 2  # Minimum length of the minor axis
    max_minor_axis = 20  # Maximum length of the minor axis

    for cnt in contours:
        if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse  # MA: major axis, ma: minor axis

            # Check if the ellipse dimensions are within the specified thresholds
            if min_major_axis <= MA <= max_major_axis and min_minor_axis <= ma <= max_minor_axis:
                center = (int(x), int(y))


                # Draw a red point at the center of the ellipse
                cv2.circle(image, center, 2, (0, 0, 255), -1)  # Red point on original image

    # Show the result
    cv2.imshow('Original Image with Ellipses', image)
    cv2.imshow('Edges with Ellipses', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






image = cv2.imread("images/cropped_board.png")
find_balls(image)