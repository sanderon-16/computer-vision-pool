from dataclasses import dataclass

import cv2
import numpy as np
from typing import Union, Tuple, List

from cv2 import Mat

Image = Union[Mat, np.ndarray]


class Ball:
    def __init__(self, position: Tuple[int, int], color: str, striped: bool):
        self.position = position
        self.striped = striped  # True if striped, False if solid
        self.color = color
        self.pocketed = False


@dataclass
class Board:
    width: int
    height: int

def find_board():
    # finds board
    return Board(112, 224)

def transform_board(image: Image, rect) -> Image:
    # Get the coordinates of the corners of the board
    x1, y1, x2, y2, x3, y3, x4, y4 = rect

    # Set the target size for the new image
    target_width = 112*4
    target_height = 224*4

    # Define the new coordinates of the corners in the new image
    new_rect = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32), new_rect)

    # Apply the perspective transformation to the original image
    transformed_image = cv2.warpPerspective(image, matrix, (target_width, target_height))

    return transformed_image


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
    rect = [817, 324, 1186, 329, 1364, 836, 709, 831]
    board = transform_board(image, rect)
    show_image(board)
    cv2.imwrite('cropped_board.png', board)


if __name__ == "__main__":
    filepath = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\downloaded_images\board_uncropped_with_stick.jpg"
    image = cv2.imread(filepath)
    main(image)
