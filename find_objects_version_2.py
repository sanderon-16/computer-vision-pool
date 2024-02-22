import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
from pool_structure import Ball, Color, WhiteBall, Cue

Image = Union[Mat, np.ndarray]

def subtract_images(image1: Image, image2: Image) -> Image:
    image2_with_neg = image1.astype(np.int32)
    image1_with_neg = image2.astype(np.int32)
    return np.abs(image2_with_neg - image1_with_neg).astype(np.uint8)


def find_circles(balls_image, contours):
    balls_center_radius = []
    image_with_circles = balls_image.copy()
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.3 < circularity <= 1.6:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius) + 2
            if radius > 11:
                cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)
                ball = (center, radius)
                balls_center_radius.append(ball)
    return balls_center_radius, image_with_circles


def find_balls(balls_image: Image, original_image: Image):
    """
    param1: image
    Find all the balls in an image and return a list of Ball objects
    return: List[Ball]
    """
    subtracted_image = subtract_images(balls_image, original_image)
    rgb, hsv, bilateral_color, gray_1 = images_formats(subtracted_image)
    mask = take_threshold(bilateral_color, 40)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(img_contours, contours, -1, Color.WHITE.value, 1)
    ball_center_radius, image_with_circles = find_circles(balls_image, contours)
    return ball_center_radius, image_with_circles, img_contours, contours


def take_threshold(bilateral_color, threshold=40):
    mask = np.any(bilateral_color > threshold, axis=-1).astype(np.uint8) * 255
    return mask


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


def find_cue(ball_image, contours):
    max_length = 0
    cue_contour = None
    for cnt in contours:
        approx, length = calculateContourLength(cnt)
        if length > max_length:
            max_length = length
            cue_contour = approx
    if cue_contour is not None:
        cv2.drawContours(ball_image, [cue_contour], -1, Color.GREEN.value, 3)
    return ball_image, cue_contour


def calculateContourLength(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    length = cv2.arcLength(approx, False)
    return approx, length


def find_lines(img_contours, image_with_circles):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 500  # maximum gap in pixels between connectable line segments
    # creating a blank to draw lines on
    line_image = np.copy(img_contours) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(img_contours, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines = [line for line in lines if abs(line[0][0] - line[0][2]) > 20 and abs(line[0][1] - line[0][3]) > 20]
    filtered_lines = []
    if lines is None:
        return filtered_lines
    for line in lines:
        if not filtered_lines:
            filtered_lines.append(line)
            continue
        for line_2 in filtered_lines:
            if (abs(line[0][0] - line_2[0][0]) < 10 and abs(line[0][2] - line_2[0][2]) < 10) or (
                    abs(line[0][1] - line_2[0][1]) < 10 and abs(line[0][3] - line_2[0][3]) < 10):
                break
    for line in filtered_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image_with_circles, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return filtered_lines, image_with_circles


def find_ball_color(image, center, radius):
    """
    #     WHITE = (255, 255, 255)
    #     YELLOW = (0, 255, 255)
    #     BLUE = (255,0,0)
    #     RED = (0,0,255)
    #     PURPLE = (255,0,255)
    #     ORANGE = (0,165,255)
    #     GREEN = (0,255,0)
    #     BROWN = (42,42,165)
    #     BLACK = (0,0,0)
    """
    sum_white = 0
    sum_red = 0
    sum_green = 0
    sum_blue = 0
    counter = 0
    stripped = False
    radius = 5
    sum_red = image[center[1]][center[0]][2]
    sum_green = image[center[1]][center[0]][1]
    sum_blue = image[center[1]][center[0]][0]

    if sum_red > 200 and sum_green > 200 and sum_blue > 200:
        return Color.WHITE, False
    if sum_white > 50:
        stripped = True
    if sum_red > 130 and sum_green < 100 and sum_blue < 100:
        return Color.RED, stripped
    if sum_red < 100 and sum_green > 130 and sum_blue < 100:
        return Color.GREEN, stripped
    if sum_red < 100 and sum_green < 100 and sum_blue > 130:
        return Color.BLUE, stripped
    if sum_red > 130 and sum_green > 130 and sum_blue < 100:
        return Color.YELLOW, stripped
    if sum_red > 100 and sum_green < 100 and sum_blue > 130:
        return Color.PURPLE, stripped
    if sum_red > 130 and sum_green > 100 and sum_blue < 100:
        return Color.ORANGE, stripped
    if sum_red < 100 and sum_green < 100 and sum_blue < 100:
        return Color.BLACK, stripped
    if sum_red < 130 and sum_green < 130 and sum_blue < 130:
        return Color.BROWN, stripped
    return Color.RED, stripped


def create_ball_objects(ball_center_radius, original_image):
    balls = []
    for center_radius in ball_center_radius:
        center = center_radius[0]
        radius = center_radius[1]
        color, stripped = find_ball_color(original_image, center, radius)
        ball = Ball(center, False, 15, color, False)
        balls.append(ball)
    cue_ball = None
    for ball in balls:
        if ball.color == Color.WHITE:
            cue_ball = WhiteBall(ball.position, False, 15, ball.color, False)
            break
    if cue_ball is None:
        cue_ball = WhiteBall(balls[0].position, False, 15, Color.WHITE, False)
    return balls, cue_ball


def create_cue_object(cue_contour, original_image, cue_ball_position):
    x1, y1, x2, y2 = cue_contour[0][0][0], cue_contour[0][0][1], cue_contour[1][0][0], cue_contour[1][0][1]
    distance_1 = np.sqrt((x1 - cue_ball_position[0]) ** 2 + (y1 - cue_ball_position[1]) ** 2)
    distance_2 = np.sqrt((x2 - cue_ball_position[0]) ** 2 + (y2 - cue_ball_position[1]) ** 2)
    if distance_1 > distance_2:
        direction = np.array([x1 - x2, y1 - y2])
        cue_edge = np.array([x1, y1])
    else:
        direction = np.array([x2 - x1, y2 - y1])
        cue_edge = np.array([x1, y1])
    cue = Cue(cue_edge, direction)
    return cue


def find_objects(balls_image: Image, original_image: Image):
    ball_center_radius, image_with_circles, img_contours, contours = find_balls(balls_image, original_image)
    image_with_circles_and_cue, cue_contour = find_cue(image_with_circles, contours)
    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    cue = create_cue_object(cue_contour, original_image, cue_ball.position)
    return balls, cue_ball, cue


def ball_objects(balls_image: Image, original_image: Image):
    ball_center_radius, image_with_circles, img_contours, contours = find_balls(balls_image, original_image)
    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    if cue_ball is None:
        return None, None
    if not balls:
        return None, None
    return balls, cue_ball


def cue_object(balls_image: Image, original_image: Image):
    ball_center_radius, image_with_circles, img_contours, contours = find_balls(balls_image, original_image)
    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    image_with_circles_and_cue, cue_contour = find_cue(image_with_circles, contours)
    if cue_contour is None:
        return None
    cue = create_cue_object(cue_contour, original_image, cue_ball.position)
    return cue


if __name__ == '__main__':
    board_image = cv2.imread(r"photos_2\WIN_20240222_09_04_29_Pro.jpg")
    balls_image = cv2.imread(r"photos_2\WIN_20240222_09_06_19_Pro.jpg")
    balls, cue_ball = ball_objects(balls_image, board_image)
    cue = cue_object(balls_image, board_image)
