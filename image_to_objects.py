from typing import List, Tuple
import cv2
import numpy as np
from .find_balls import find_balls, Image
from .find_cue import find_cue
from .pool_structure import WhiteBall, Color, Ball, Board, Cue


def find_cue_ball_center(centers: Tuple[int], image: Image):
    """
    This function pass on radius of every ball, and find what is most white
    :param centers:
    :return:
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    max_white_ball = 0
    white_ball = 0
    for center in centers:
        h_avarage = 0
        s_avarage = 0
        v_avarage = 0
        x, y = center
        radius = 10
        # find the sum of white in the area
        for x in range(x-10, x+10, 1):
            if x<0 or x>=len(image[0]):
                continue
            for y in range(y-10, y+10, 1):
                if y<0 or y>=len(image):
                    continue
                h_avarage += hsv[y][x][2]/100
                s_avarage += hsv[y][x][1]/100
                v_avarage += hsv[y][x][0]/100

        if h_avarage + s_avarage + v_avarage > max_white_ball:
            max_white_ball = h_avarage + s_avarage + v_avarage
            white_ball_center = center
    #mark the white ball on the image and the word cue ball
    image_with_cue_ball = cv2.circle(image, white_ball_center, 10, (0, 255, 0), 2)
    cv2.putText(image_with_cue_ball, "Cue Ball", (white_ball_center[0], white_ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return white_ball_center

def create_ball_objects(centers, cue_ball_center, image_with_balls):
    balls = []
    for center in centers:
        center_np = np.array(center)
        print(center, cue_ball_center)
        if center == cue_ball_center:
            white_ball = WhiteBall(center_np, False, 15, Color.WHITE, False)
            balls.append(white_ball)
            continue
        color = find_ball_color(center, image_with_balls)
        balls.append(Ball(center_np, False, 15, color, False))
    return balls, white_ball

def create_balls(image: Image):
    centers, image_with_balls = find_balls(image)
    cv2.imwrite(r"images\balls1.png", image_with_balls)
    cue_ball_center = find_cue_ball_center(centers, image)
    balls, white_ball = create_ball_objects(centers, cue_ball_center, image_with_balls)
    return balls, white_ball

def find_ball_color(center, image: Image):
    r_avarage = 0
    g_avarage = 0
    b_avarage = 0

    for x in range(center[0] - 10, center[0] + 10, 1):
        if x < 0 or x >= len(image[0]):
            continue
        for y in range(center[1] - 10, center[1] + 10, 1):
            if y < 0 or y >= len(image):
                continue
            r_avarage += image[y][x][2] / 100
            g_avarage += image[y][x][1] / 100
            b_avarage += image[y][x][0] / 100
    most_close_color = Color.RED
    min_dis_from_color = 10000000000000
    for color in Color:
        dis_from_color = np.sqrt((color.value[2] - r_avarage) ** 2 + (color.value[1] - g_avarage) ** 2 + (color.value[0] - b_avarage) ** 2)
        if dis_from_color < min_dis_from_color:
            most_close_color = color
            min_dis_from_color = dis_from_color
    return most_close_color

def find_objects(image: Image):
    board = Board(len(image[0]), len(image))
    balls, cue_ball = create_balls(image)
    cue_edge, direction = find_cue(image, cue_ball.position)
    if cue_edge == None:
        return board, balls, cue_ball, None
    cue_edge_np = np.array(cue_edge)
    direction_np = np.array(direction)
    cue = Cue(cue_edge_np, direction_np)
    return board, balls, cue_ball, cue


image = cv2.imread(r"output_images\cropped_board_1.png")
cv2.imwrite(r"images\original1.png", image)

