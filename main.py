import cv2
import numpy as np
from typing import Union, Tuple, List

from cv2 import Mat

Image = Union[Mat, np.ndarray]


class Ball:
    def __init__(self, position: Tuple[float, float], color: str, striped: bool):
        self.position = position
        self.striped = striped  # True if striped, False if solid
        self.color = color
        self.pocketed = False


def find_board(image: Image) -> Image:
    """
    param1: image
    Find a pool board from an image and return the image that contains the board in correct dimensions
    return: image
    """
    # Define the lower and upper bounds of the green color in HSV
    lower_green = np.array([50, 110, 50])
    upper_green = np.array([80, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    to_show = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0))
    show_image(to_show)

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to be with fewer details
    epsilon = 0.0055 * cv2.arcLength(max_contour, True) # TODO fix this huge magic number
    max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

    # generate convex hull:
    convex_hull = cv2.convexHull(max_contour)

    # Generate a blank black image with only the convex_hull drawn on it
    blank_image = np.zeros_like(image)
    convex_hull_image = cv2.drawContours(blank_image.copy(), [convex_hull], -1,(255, 255, 255), 2)

    # You can also draw the convex hull on the original image for visualization
    image_with_convex_hull = image.copy()
    cv2.drawContours(image_with_convex_hull, [convex_hull], -1, (0, 255, 0), 2)

    # Display the images (you can use your own function for displaying)
    show_image(image_with_convex_hull)
    show_image(convex_hull_image)

    # convert convex_hull_image type to CV_8UC1
    convex_hull_image_grayscale = cv2.cvtColor(convex_hull_image, cv2.COLOR_BGR2GRAY)

    # Find the lines using HoughLines on the binary image
    lines = cv2.HoughLines(convex_hull_image_grayscale, 1, np.pi / 180, 50)

    # Extract four most prominent lines
    lines = lines[:4]
    image_with_lines = image.copy()
    # Get the intersections of the lines
    intersections = []
    for i in range(len(lines) - 1):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]

            # Calculate the intersection point
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                          [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            intersection = np.linalg.solve(A, b)
            intersection = tuple(map(int, intersection))
            intersections.append(intersection)

    # Draw the intersections on the image_with_lines
    for point in intersections:
        cv2.circle(image_with_lines, point, 5, (0, 0, 255), -1)

    # Draw the lines on image_with_lines
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the image_with_lines
    show_image(image_with_lines)

    # Perform perspective transformation to correct the perspective
    src_pts = max_contour.reshape(4, 2).astype(np.float32)
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(image, matrix, (w, h))

    # Resize the corrected image to the correct dimensions (2x1 proportional)
    new_width = int(2 * h)
    resized_image = cv2.resize(corrected_image, (new_width, h))

    return resized_image


def find_balls(image: Image) -> List[Ball]:
    """
    param1: image
    Find all balls in an image and return a list of balls with their positions
    return: list
    """
    pass


def find_cue(image: Image) -> List[Tuple[float, float]]:
    """
    param1: image
    Find the cue in an image and return the position of the cue
    return: list
    """
    pass


def show_image(image: Image):
    # displays an image:
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image: Image):
    board = find_board(image)
    show_image(board)


if __name__ == "__main__":
    filepath = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
    image = cv2.imread(filepath)
    main(image)
