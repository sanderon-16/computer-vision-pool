import cv2
import numpy as np


def find_balls(blank_table_img_path, table_with_balls_img_path, output_img_path):
    # Load the images
    blank_table_img = cv2.imread(blank_table_img_path)
    table_with_balls_img = cv2.imread(table_with_balls_img_path)

    # Ensure both images have the same dimensions
    if blank_table_img.shape != table_with_balls_img.shape:
        raise ValueError("Image dimensions do not match")

    # Convert the images to grayscale
    blank_table_gray = cv2.cvtColor(blank_table_img, cv2.COLOR_BGR2GRAY)
    table_with_balls_gray = cv2.cvtColor(table_with_balls_img, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the two images
    diff_img = cv2.absdiff(blank_table_gray, table_with_balls_gray)

    # Apply thresholding to find the regions with balls
    _, thresholded_diff = cv2.threshold(diff_img, 37, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store ball positions
    ball_positions = []

    # Draw circles around the detected balls on the original image
    result_img = table_with_balls_img.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # You may need to adjust this threshold
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(result_img, center, radius, (0, 0, 255), 2)  # Red circle

            # Store the center coordinates of the detected ball as a tuple (x, y)
            ball_positions.append((center[0], center[1]))

    # Save or display the resulting image
    cv2.imwrite(output_img_path, result_img)

    # Return the list of ball positions
    return ball_positions


# Example usage:
ball_positions = find_balls("blank_table.jpg", "pool_table_with_balls.jpg", "marked_table.jpg")
print("List of Ball Positions:", ball_positions)
