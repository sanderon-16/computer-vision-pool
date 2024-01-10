import cv2
import numpy as np
from typing import Union

Image = Union[Mat, n.ndarray]


class Ball:
    def __init__(self, position: Tuple(float, float), color: str, striped: bool):
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
    pass


def find_balls(image: Image) -> list[Ball]:
    """
    param1: image
    Find all balls in an image and return a list of balls with their positions
    return: list
    """
    pass


def find_cue(image: Image) -> List[float, float]:
    """
    param1: image
    Find the cue in an image and return the position of the cue
    return: list
    """
    pass


def main(image: Image):
    pass


if __name__ == "__main__":
    image = None
    main(image)
