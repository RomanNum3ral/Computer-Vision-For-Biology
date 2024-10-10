import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Function to process each image
def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define color range
    lower_bound = np.array([50, 70, 30], dtype=np.uint8)
    upper_bound = np.array([150, 255, 150], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    
    # Create an output 
    output_image = np.zeros_like(image_rgb)
    output_image[mask != 0] = (255, 255, 255)
    
    # Convert to grayscale
    output_gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
    
    # Convert grayscale image to binary
    _, binary = cv2.threshold(output_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Define output filename
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{filename}_green_yellow.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save processed image
    cv2.imwrite(output_path, binary)
    print(f"Processed: {image_path} -> {output_path}")

# Function to process all images in folder
def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)

# Set input and output folders
input_folder = "in"
output_folder = "out"

# Run the processing
process_images_in_folder(input_folder, output_folder)