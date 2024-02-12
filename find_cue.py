import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
Image = Union[Mat, np.ndarray]



def images_formats(image: Image) -> Tuple[Image]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bilateral_color = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)
    return image, rgb, hsv, bilateral_color, gray

def find_cue(image: Image, white_ball_coordinate) -> List[Tuple[int]]:
    """
    :param image: Image
    :return: List of tuples containing the coordinates of the cue
    """

    cut_and_resized_image, rgb, hsv, bilateral_color, gray = images_formats(image)
    lines = find_lines(gray)
    if len(lines) > 1:
        print("more than one line found")
        return
    if len(lines) == 0:
        print("No lines found")
        return
    line = lines[0]
    cue_parameters = return_cue_parameters(gray, line, white_ball_coordinate)
    return cue_parameters





def find_lines(gray_image: Image) -> List[Tuple[int]]:
    """
    :param image: Image
    :return: List of tuples containing the coordinates of the lines
    """
    edges = cv2.Canny(gray_image, 50, 100)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # creating a blank to draw lines on
    line_image = np.copy(gray_image) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines = [line for line in lines if abs(line[0][0] - line[0][2]) > 20 and abs(line[0][1] - line[0][3]) > 20]
    filtered_lines = []
    for line in lines:
        if not filtered_lines:
            filtered_lines.append(line)
            continue
        for line_2 in filtered_lines:
            if (abs(line[0][0] - line_2[0][0]) < 10 and abs(line[0][2] - line_2[0][2]) < 10) or (abs(line[0][1] - line_2[0][1]) < 10 and abs(line[0][3] - line_2[0][3]) < 10):
                break
    for line in filtered_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    cv2.imwrite(r"images\cue1.png", line_image)
    return filtered_lines


def return_cue_parameters(gray_image: Image, line: List[Tuple[int]], white_ball_coordinate) -> Tuple[int, int]:
    """
    :param gray_image: Image
    :param lines: List of tuples containing the coordinates of the lines
    :return: Tuple containing the coordinates of the cue
    """
    x1, y1, x2, y2 = line[0]
    x_ball = white_ball_coordinate[0]
    y_ball = white_ball_coordinate[1]
    dis_1 = np.sqrt((x1 - x_ball) ** 2 + (y1 - y_ball) ** 2)
    dis_2 = np.sqrt((x2 - x_ball) ** 2 + (y2 - y_ball) ** 2)
    if dis_1 < dis_2:
        return x1, (x1-x2, y1-y2)
    return x2, (x2-x1, y2-y1)

