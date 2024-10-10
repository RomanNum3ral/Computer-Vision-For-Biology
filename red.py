import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define colors for different categories
colors = {
    0: [0, 0, 0],     # Black
    2: [255, 255, 255], # White
    3: [255, 0, 0]    # Red
}

# Function to process each image
def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    
    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define color ranges
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([50, 50, 50], dtype=np.uint8)
    
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    
    lower_red = np.array([100, 0, 0], dtype=np.uint8)
    upper_red = np.array([255, 50, 50], dtype=np.uint8)
    
    # Create color masks
    mask_black = cv2.inRange(image_rgb, lower_black, upper_black)
    mask_white = cv2.inRange(image_rgb, lower_white, upper_white)
    mask_red = cv2.inRange(image_rgb, lower_red, upper_red)
    
    # Create an output mask where black == 0, white == 2, red == 3
    output_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    output_mask[mask_black != 0] = 0
    output_mask[mask_white != 0] = 2
    output_mask[mask_red != 0] = 3
    
    # Define output filename
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{filename}_color_mask_test.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save image
    cv2.imwrite(output_path, output_mask)
    print(f"Processed: {image_path} -> {output_path}")
    
    # Create RGB representation of the mask
    rgb_mask = np.zeros((output_mask.shape[0], output_mask.shape[1], 3), dtype=np.uint8)
    for label, color in colors.items():
        rgb_mask[output_mask == label] = color
    
    # Display the RGB mask
    plt.figure(figsize=(6, 6))
    plt.title('RGB Color Mask')
    plt.imshow(rgb_mask)
    plt.axis('off')  # Hide the axis
    plt.show()

# Process images in folder
def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)

input_folder = "in"
output_folder = "out"

process_images_in_folder(input_folder, output_folder)