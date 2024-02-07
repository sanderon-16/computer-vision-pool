import os
import cv2
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from functools import partial
from image_processing import generate_projection

class RectAdjustmentApp:
    def __init__(self, image_path, rect):
        self.image = cv2.imread(image_path)
        self.cropped_image = None

        if self.image is None:
            raise ValueError(f"Error loading image from path: {image_path}")

        self.rect = rect
        self.selected_corner = None
        self.scale_factor = 0.5

        self.root = tk.Tk()
        self.root.title("Rectangle Adjustment")

        # Lock GUI size to the size of the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        self.canvas_original = tk.Canvas(self.root, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas_original.pack(side=tk.LEFT, padx=10, pady=10)

        # Label to display rectangle parameters
        self.label_var = tk.StringVar()
        self.label_var.set(f"Rectangle Parameters: {[x/self.scale_factor for x in self.rect]}")
        self.label = tk.Label(self.root, textvariable=self.label_var)
        self.label.pack(side=tk.TOP, pady=10)

        # Calculate maximum image size based on 2/5 of the window size
        max_width = int(self.root.winfo_screenwidth() * 2 / 5)
        max_height = int(self.root.winfo_screenheight() * 4 / 5)

        # Determine the scaling factor
        scale_factor_width = max_width / self.image.shape[1]
        scale_factor_height = max_height / self.image.shape[0]
        self.scale_factor = min(scale_factor_width, scale_factor_height)

        # Bind mouse click event to canvas
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)

        # Bind arrow key events to canvas
        self.root.bind("<Left>", self.on_left_arrow)
        self.root.bind("<Right>", self.on_right_arrow)
        self.root.bind("<Up>", self.on_up_arrow)
        self.root.bind("<Down>", self.on_down_arrow)

        # Create a button for saving the image
        self.save_button = tk.Button(self.root, text="Save Rect", command=self.save_rect)
        self.save_button.pack(side=tk.TOP, pady=10)

        # Create a button for loading a new image
        self.load_button = tk.Button(self.root, text="Load New Image", command=self.load_new_image)
        self.load_button.pack(side=tk.TOP, pady=10)

        # Create a canvas for displaying the live webcam feed
        self.canvas_webcam = tk.Canvas(self.root, width=max_width, height=max_height)
        self.canvas_webcam.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize webcam
        self.cap = cv2.VideoCapture(1)  # TODO make webcam 0 work

        # # Create a window for displaying the cropped image
        self.cropped_image_window = tk.Toplevel(self.root)
        self.initialize_cropped_image_window()

        # Call the draw_rect and update_webcam functions periodically
        self.draw_rect()
        self.update_webcam()

    def initialize_cropped_image_window(self):
        # Set the window attributes
        self.cropped_image_window.attributes('-borderless', True)
        # self.cropped_image_window.attributes('-fullscreen', True)  # Set to fullscreen
        # self.cropped_image_window.attributes('-topmost', True)  # Bring to front
        # self.cropped_image_window.attributes('-alpha', 0.7)  # Set transparency (adjust as needed)

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set initial position for the second screen (adjust as needed)
        second_screen_x = screen_width  # X-coordinate for the second screen
        second_screen_y = 0  # Y-coordinate for the second screen

        # Set the initial position of the window
        self.cropped_image_window.geometry(f"+{second_screen_x}+{second_screen_y}")

        # Create a canvas for displaying the cropped image
        self.cropped_image_canvas = tk.Canvas(self.cropped_image_window)
        self.cropped_image_canvas.pack(fill=tk.BOTH, expand=tk.YES)

    def draw_rect(self):
        # Draw the original image on the original canvas
        self.transform_and_display()
        # Update the Tkinter window
        self.root.update()

        # Update the label with the current rectangle parameters
        self.label_var.set(f"Rectangle Parameters: {[int(x / self.scale_factor) for x in self.rect]}")

        # Display the cropped_image in the borderless and fullscreen window
        if self.cropped_image is not None:
            cropped_image_rgb = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            img_tk_cropped = ImageTk.PhotoImage(image=Image.fromarray(cropped_image_rgb))
            self.cropped_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk_cropped)
            self.cropped_image_canvas.image = img_tk_cropped

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
        self.transform_and_display()

    def on_right_arrow(self, event):
        self.adjust_selected_corner(1, 0)
        self.transform_and_display()

    def on_up_arrow(self, event):
        self.adjust_selected_corner(0, -1)
        self.transform_and_display()

    def on_down_arrow(self, event):
        self.adjust_selected_corner(0, 1)
        self.transform_and_display()

    def adjust_selected_corner(self, delta_x, delta_y):
        if self.selected_corner is not None:
            self.rect[self.selected_corner] += delta_x
            self.rect[self.selected_corner + 1] += delta_y

            # Update the label with the new rectangle parameters
            self.label_var.set(f"Rectangle Parameters: {self.rect}")

    def transform_and_display(self):
        # Transform the image using the specified rectangle
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        transformed_image = generate_projection(self.image, actual_rect)
        # Display the transformed image on the canvas

        image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk_transformed = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=img_tk_transformed)
        self.canvas_original.image = img_tk_transformed
        self.cropped_image = transformed_image

    def update_webcam(self):
        # Capture frame from the webcam
        ret, frame = self.cap.read()
        max_width = int(self.root.winfo_screenwidth() * 2 / 5)
        max_height = int(self.root.winfo_screenheight() * 4 / 5)
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to match the canvas size
            frame_resized = cv2.resize(frame_rgb, (max_width, max_height))

            # Convert the frame to ImageTk format
            img_tk_webcam = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

            # Display the frame on the canvas
            self.canvas_webcam.create_image(0, 0, anchor=tk.NW, image=img_tk_webcam)
            self.canvas_webcam.image = img_tk_webcam

        # Call the update_webcam function again after a delay (in milliseconds)
        self.root.after(100, self.update_webcam)

    def save_rect(self):
        print(self.rect)

    def load_new_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                raise ValueError(f"Error loading image from path: {file_path}")

            self.rect = [int(self.scale_factor*x) for x in [817, 324, 1186, 329, 1364, 836, 709, 831]]  # Reset rectangle coordinates
            self.selected_corner = None

            # Update the label with the new rectangle parameters
            self.label_var.set(f"Rectangle Parameters: {self.rect}")


if __name__ == '__main__':
    image_path = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
    initial_rect = [int(0.4*x) for x in [817, 324, 1186, 329, 1364, 836, 709, 831]] # Initial rectangle coordinates

    try:

        app = RectAdjustmentApp(image_path, initial_rect)
        app.root.mainloop()
    except ValueError as e:
        print(f"Error: {e}")
