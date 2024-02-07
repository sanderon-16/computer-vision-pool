import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
from image_processing import transform_board

class RectAdjustmentApp:
    def __init__(self, image_path, rect):
        self.image = cv2.imread(image_path)
        self.cropped_image = None

        if self.image is None:
            raise ValueError(f"Error loading image from path: {image_path}")

        self.rect = rect
        self.selected_corner = None

        self.root = tk.Tk()
        self.root.title("Rectangle Adjustment")

        # Lock GUI size to the size of the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        self.canvas_original = tk.Canvas(self.root, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.entry_vars = [tk.StringVar() for _ in range(8)]
        for i in range(8):
            entry = ttk.Entry(self.root, textvariable=self.entry_vars[i], width=6)
            entry.insert(0, str(self.rect[i]))
            entry.pack()

        # Label to display rectangle parameters
        self.label_var = tk.StringVar()
        self.label_var.set(f"Rectangle Parameters: {self.rect}")
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

        # Create a button for applying transformation
        self.transform_button = tk.Button(self.root, text="Transform Image", command=self.transform_and_display)
        self.transform_button.pack(side=tk.TOP, pady=10)

        # Create a button for saving the image
        self.save_button = tk.Button(self.root, text="Save Image", command=self.save_image)
        self.save_button.pack(side=tk.TOP, pady=10)

        # Create a button for loading a new image
        self.load_button = tk.Button(self.root, text="Load New Image", command=self.load_new_image)
        self.load_button.pack(side=tk.TOP, pady=10)

        # Create a canvas for displaying the transformed image
        self.canvas_transformed = tk.Canvas(self.root, width=max_width, height=max_height)
        self.canvas_transformed.pack(side=tk.LEFT, padx=10, pady=10)

        # Initialize counter for saved images
        self.counter = 1

        # Call the draw_rect function periodically
        self.draw_rect()

    def draw_rect(self):
        # Draw the original image on the original canvas
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_original.image = img_tk

        # Draw the rectangle on the original canvas
        self.canvas_original.create_polygon(self.rect[0], self.rect[1], self.rect[2], self.rect[3],
                                            self.rect[4], self.rect[5], self.rect[6], self.rect[7], outline="red", fill="")

        # Update the Tkinter window
        self.root.update()

        # Update the label with the current rectangle parameters
        self.label_var.set(f"Rectangle Parameters: {self.rect}")

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
        transformed_image = transform_board(self.image, actual_rect)
        # Display the transformed image on the canvas

        image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk_transformed = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas_transformed.create_image(0, 0, anchor=tk.NW, image=img_tk_transformed)
        self.canvas_transformed.image = img_tk_transformed
        self.cropped_image = transformed_image

    def save_image(self):
        base_directory = os.getcwd()  # Get the current working directory
        output_subdirectory = "output_images"  # Subdirectory for saving images

        # Create the output directory if it doesn't exist
        output_directory = os.path.join(base_directory, output_subdirectory)
        os.makedirs(output_directory, exist_ok=True)

        # Save the image with a unique name based on the counter
        image_name = f'cropped_board_{self.counter}.png'
        image_path = os.path.join(output_directory, image_name)
        cv2.imwrite(image_path, self.cropped_image)

        # Increment the counter
        self.counter += 1

    def load_new_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                raise ValueError(f"Error loading image from path: {file_path}")

            self.rect = [50, 50, 250, 50, 250, 200, 50, 200]  # Reset rectangle coordinates
            self.selected_corner = None

            # Update the label with the new rectangle parameters
            self.label_var.set(f"Rectangle Parameters: {self.rect}")


# Example usage:
# Replace "your_image.jpg" with the path to your actual image file
if __name__ == '__main__':
    image_path = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
    initial_rect = [100, 100, 500, 100, 500, 600, 100, 600]  # Initial rectangle coordinates

    try:
        app = RectAdjustmentApp(image_path, initial_rect)
        app.root.mainloop()
    except ValueError as e:
        print(f"Error: {e}")
