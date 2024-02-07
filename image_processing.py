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

class Board:
    width = 112*4
    height = 224*4
    def __init(self, balls: List[Ball]):
        self.balls = balls


def transform_board(image: Image, rect) -> Image:
    # Get the coordinates of the corners of the board
    x1, y1, x2, y2, x3, y3, x4, y4 = rect

    # Set the target size for the new image
    target_width = Board.width
    target_height = Board.height

    # Define the new coordinates of the corners in the new image
    new_rect = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32), new_rect)

    # Apply the perspective transformation to the original image
    transformed_image = cv2.warpPerspective(image, matrix, (target_width, target_height))

    return transformed_image

def generate_projection(image_to_project: Image, rect) -> Image:
    x1, y1, x2, y2, x3, y3, x4, y4 = rect

    # Define the target rectangle dimensions
    target_width = 1920
    target_height = 1080

    # Define the source rectangle (coordinates of the four corners)
    src_pts = np.float32(
        [[0, 0], [image_to_project.shape[1], 0], [image_to_project.shape[1], image_to_project.shape[0]],
         [0, image_to_project.shape[0]]])

    # Define the target rectangle (coordinates of the four corners)
    target_pts = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))

    # Apply the perspective transformation to the image
    projected_image = cv2.warpPerspective(image_to_project, matrix, (target_width, target_height),
                                          borderValue=(0, 0, 0))

    return projected_image

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
    board = generate_projection(image, rect)
    show_image(board)
    cv2.imwrite('proj.png', board)


if __name__ == "__main__":
    filepath = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\output_images\cropped_board.png"
    image = cv2.imread(filepath)
    main(image)
