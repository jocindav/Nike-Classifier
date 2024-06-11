import os
import numpy as np
from PIL import Image
import torch
from NikeNeuralNetwork import NikeNeuralNetwork

def preprocess(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(128 * 128)
    return image_array

# def load_data(data_dir, types):
#     images = []
#     labels = []
#     for label, category in enumerate(types):
#         category_dir = os.path.join(data_dir, category)
#         print(f"Checking directory: {category_dir}")
#         if not os.path.exists(category_dir):
#             print(f"Directory {category_dir} does not exist.")
#             continue
#         for filename in os.listdir(category_dir):
#             file_path = os.path.join(category_dir, filename)
#             if not os.path.isfile(file_path):
#                 continue
#             try:
#                 img = preprocess(file_path)
#                 images.append(img)
#                 labels.append(label)
#             except (IOError, ValueError) as e:
#                 print(f"Skipping file {file_path}: {e}")
#                 continue
#     return np.array(images), np.array(labels)

def load_data(data_dir, types):
    images = []
    labels = []
    num_classes = len(types)
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
                one_hot_label = [0] * num_classes
                one_hot_label[label] = 1
                labels.append(one_hot_label)
            except (IOError, ValueError) as e:
                print(f"Skipping file {file_path}: {e}")
                continue
    return np.array(images), np.array(labels).astype(float)

# Define directories and types
data_dir = "datasets/nikeTrain"
types = ["Nike", "Adidas", "Puma"]

# Load data
images, labels = load_data(data_dir, types)

# Initialize and train the model
model = NikeNeuralNetwork()
model.train_model(images, labels, epochs=3)

# Save the trained model
torch.save(model.state_dict(), "Nike.pth")
