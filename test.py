import os
import numpy as np
import torch
from NikeNeuralNetwork import NikeNeuralNetwork
from PIL import Image

def preprocess(image_path):
    image = Image.open(image_path).convert('L')  #  grayscale
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(128 * 128)
    return image_array

def load_data(data_dir, types):
    images = []
    labels = []
    for label, category in enumerate(types):
        category_dir = os.path.join(data_dir, category)
        print(f"Checking directory: {category_dir}")
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} does not exist.")
            continue
        for filename in os.listdir(category_dir):
            file_path = os.path.join(category_dir, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                img = preprocess(file_path)
                images.append(img)
                labels.append(label)
            except (IOError, ValueError) as e:
                print(f"Skipping file {file_path}: {e}")
                continue
    return np.array(images), np.array(labels)

# Define directories and types
data_dir = "datasets/nikeTest"
types = ["Nike", "Adidas", "Puma"]

# Load data
images, labels = load_data(data_dir, types)

if len(images) == 0:
    raise ValueError("No images found in the dataset. Please check your data directory and ensure it contains valid images.")

# Initialize the model and load the trained weights
model = NikeNeuralNetwork()
model.load_state_dict(torch.load("Nike.pth"))

# Evaluate the model
correct = 0
total = len(labels)
for img, label in zip(images, labels):
    output = model.forward(img).detach().numpy()
    guess = np.argmax(output)
    if guess == label:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
