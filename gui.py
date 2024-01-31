import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class RectAdjustmentApp:
    def __init__(self, image_path, rect):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Error loading image from path: {image_path}")

        self.rect = rect
        self.selected_corner = None

        self.root = tk.Tk()
        self.root.title("Rectangle Adjustment")

        self.canvas = tk.Canvas(self.root, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Bind mouse click event to canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Bind arrow key events to canvas
        self.root.bind("<Left>", self.on_left_arrow)
        self.root.bind("<Right>", self.on_right_arrow)
        self.root.bind("<Up>", self.on_up_arrow)
        self.root.bind("<Down>", self.on_down_arrow)

        # Call the draw_rect function periodically
        self.draw_rect()

    def draw_rect(self):
        # Draw the image on the canvas
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        scale_factor = 0.7
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

        # Draw the rectangle on the canvas
        self.canvas.create_polygon(self.rect[0], self.rect[1], self.rect[2], self.rect[3], self.rect[4], self.rect[5], self.rect[6], self.rect[7], outline="red", fill="")

        # Update the Tkinter window
        self.root.update()

        # Call the draw_rect function again after a delay (in milliseconds)
        self.root.after(100, self.draw_rect)

    def on_canvas_click(self, event):
        min_distance = float('inf')
        selected_corner = None

        # Calculate distance between mouse click position and each corner
        for i in range(0, len(self.rect), 2):
            x, y = self.rect[i], self.rect[i + 1]
            distance = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                selected_corner = i

        # Select the corner with the smallest distance
        self.selected_corner = selected_corner
    def on_left_arrow(self, event):
        self.adjust_selected_corner(-1, 0)

    def on_right_arrow(self, event):
        self.adjust_selected_corner(1, 0)

    def on_up_arrow(self, event):
        self.adjust_selected_corner(0, -1)

    def on_down_arrow(self, event):
        self.adjust_selected_corner(0, 1)

    def adjust_selected_corner(self, delta_x, delta_y):
        if self.selected_corner is not None:
            self.rect[self.selected_corner] += delta_x
            self.rect[self.selected_corner + 1] += delta_y

# Example usage:
# Replace "your_image.jpg" with the path to your actual image file
image_path = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
initial_rect = [50, 50, 250, 50, 250, 200, 50, 200]  # Initial rectangle coordinates

try:
    app = RectAdjustmentApp(image_path, initial_rect)
    app.root.mainloop()
except ValueError as e:
    print(f"Error: {e}")
