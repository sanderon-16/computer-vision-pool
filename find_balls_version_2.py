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


def find_circles(balls_image, contours):

    for cnt in contours:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(cnt, True)

        # Calculate the area of the contour
        area = cv2.contourArea(cnt)

        # Avoid division by zero and find the circularity factor
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Check if the contour is circular (circularity close to 1 is ideal for a circle)
        if 0.3 < circularity <= 1.6:
            # Find the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)+2
            # Draw the circle on the image
            if radius > 8:
                cv2.circle(balls_image, center, radius, (0, 255, 0), 2)
    cv2.imshow('Circles', balls_image)
    cv2.waitKey(0)



def get_edge_image(balls_image: Image, original_image: Image) -> Union[List[Tuple[int]], Image]:
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    subtracted_image = subtract_images(balls_image, original_image)
    rgb, hsv, bilateral_color, gray_1 = images_formats(subtracted_image)
    mask = np.any(bilateral_color > 40, axis=-1).astype(np.uint8) * 255
    cv2.imshow('mask', mask)



    # erosion and dilation
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.imshow('mask after erosion and dilation', mask)




    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', img_contours)
    find_circles(balls_image, contours)
    cv2.waitKey(0)
    return img_contours



def pixel_to_white(image: Image):
    # pass over the pixels, and every pixel that is no black or white, will be white, and the others black
    for x in range(len(image)):
        for y in range(len(image[0])):
            if image[x][y][0] >= 60 or image[x][y][1] >= 60 or image[x][y][2] >= 60:
                image[x][y] = [255, 255, 255]
            else:
                image[x][y] = [0, 0, 0]
    return image



def find_circle(image: Image):
    pass

def find_cue():
    pass


def find_balls(edge_image: Image):
    pass


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







board_image = cv2.imread(r"photos_1\blank_board.jpg")
balls_image = cv2.imread(r"photos_1\6.jpg")
edge_image = get_edge_image(balls_image, board_image)
find_balls(edge_image)