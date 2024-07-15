import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import Image
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch

# Define the U-Net model architecture (similar to what's in the article)
# Ensure to load the model and its weights as per your training script

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Define your preprocessing steps (resize, normalize, etc.)
    transform = transforms.Compose([transforms.ToPILImage(),
 	    transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	    transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the mask using the trained U-Net model
def predict_mask(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0)
    print(predicted_mask)
    return predicted_mask.numpy()

# Function to display the original image and its predicted mask
def display_image_with_mask(image_path, mask):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    print(mask.shape)
    plt.imshow(mask)
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

# Main function to run the script
def main():
    # Load your trained model
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)

    # Accept user input for image path
    image_path = input("Enter path to the image: ")

    # Preprocess the input image
    image = preprocess_image(image_path)
    # predicted_mask = Image.open("c:\\Users\\anon\\Desktop\\Summer Research-COPY\\unet\\dataset\\train\\masks\\0aab0afa9c.png").convert("RGB")
    # Predict the mask
    predicted_mask = predict_mask(model, image)

    # Display the image with its predicted mask
    display_image_with_mask(image_path, predicted_mask)

main()