import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple

Image = Union[Mat, np.ndarray]


class Ball:
    def __init__(self, position: Tuple[int, int], color: str, striped: bool):
        self.position = position
        self.striped = striped  # True if striped, False if solid
        self.color = color
        self.radius = 13
        self.pocketed = False

#
# def find_board(image: Image) -> Image:
#     """
#     param1: image
#     Find a pool board from an image and return the image that contains the board in correct dimensions
#     return: image
#     """
#     pass
#
#
# def find_contours(board: Image, image: Image) -> List:
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_blank_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
#     diff = cv2.absdiff(gray_image, gray_blank_board)
#     cv2.imwrite(r"images\diff.jpg", diff)
#     _, thresh = cv2.threshold(diff, 58, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((8, 5), np.uint8)
#     thresh = cv2.dilate(thresh, kernel, iterations=1)
#     thresh = cv2.erode(thresh, kernel, iterations=1)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
#     return filtered_contours
#
# def find_ball_color(image: Image, ball: Ball) -> str:
#     """
#     param1: image
#     param2: ball
#     Find the color of a ball in an image and return the color
#     return: str
#     """
#     #find the dominant color in the ball
#     x, y = ball.position
#     radius = ball.radius
#
#
#
# def find_is_ball_striped(image: Image, ball: Ball) -> bool:
#     """
#     param1: image
#     param2: ball
#     Find if a ball is striped in an image and return True if striped, False if solid
#     return: bool
#     """
#     pass
#
#
# def create_ball_list(image: Image, filtered_contours: List) -> List[Ball]:
#     balls = []
#     for contour in filtered_contours:
#         # Calculate the centroid of the contour
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             x = int(M["m10"] / M["m00"])
#             y = int(M["m01"] / M["m00"])
#             # Draw a circle around the detected ball
#             cv2.circle(image, (x, y), 13, (0, 255, 0), 2)  # Green circle
#             ball = Ball(position=(x, y), color="unknown", striped=False)
#             # ball.color = find_ball_color(image, ball)
#             # ball.striped = find_is_ball_striped(image, ball)
#             balls.append(ball)
#     return balls
#
#
# def find_balls(board: Image, image: Image) -> List[Ball]:
#     """
#     param1: image
#     Find all balls in an image and return a list of balls with their positions
#     return: list
#     """
#     filtered_contours = find_contours(board, image)
#     balls = create_ball_list(image, filtered_contours)
#     show_image(image)
#     return balls
#
#
# def find_cue(image: Image) -> List[Tuple[float, float]]:
#     """
#     param1: image
#     Find the cue in an image and return the position of the cue
#     return: list
#     """
#     pass
#
#
# def show_image(image: Image):
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # Apply thresholding to find the regions with balls
#     _, thresholded_diff = cv2.threshold(diff_img, 37, 255, cv2.THRESH_BINARY)
#
#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Initialize a list to store ball positions
#     ball_positions = []
#
#     # Draw circles around the detected balls on the original image
#     result_img = table_with_balls_img.copy()
#     for contour in contours:
#         if cv2.contourArea(contour) > 100:  # You may need to adjust this threshold
#             (x, y, w, h) = cv2.boundingRect(contour)
#             center = (x + w // 2, y + h // 2)
#             radius = max(w, h) // 2
#             cv2.circle(result_img, center, radius, (0, 0, 255), 2)  # Red circle
#
#             # Store the center coordinates of the detected ball as a tuple (x, y)
#             ball_positions.append((center[0], center[1]))
#
#     # Save or display the resulting image
#     cv2.imwrite(output_img_path, result_img)
#
#     # Return the list of ball positions
#     return ball_positions
#
# # Example usage:
# ball_positions = find_balls("blank_table.jpg", "pool_table_with_balls.jpg", "marked_table.jpg")
# print("List of Ball Positions:", ball_positions)