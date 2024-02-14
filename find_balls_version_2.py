import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
Image = Union[Mat, np.ndarray]




def find_balls(balls_image: Image, original_image: Image) -> Union[List[Tuple[int]], Image]:
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    subtracted_image = cv2.subtract(balls_image, original_image)
    rgb, hsv, bilateral_color, gray = images_formats(subtracted_image)
    yiq = bgr_to_yiq(bilateral_color)
    yiq_gray = yiq[:, :, 1]


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




def images_formats(image: Image) -> Tuple[Image, Image, Image, Image]:
    """
    param1: image
    return: Tuple[Image, Image, Image, Image]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bilateral_color = cv2.bilateralFilter(rgb, 9, 75, 75)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)
    return rgb, hsv, bilateral_color, gray

