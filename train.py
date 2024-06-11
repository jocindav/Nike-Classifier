import os
import numpy as np
from PIL import Image
import torch
from NikeNeuralNetwork import NikeNeuralNetwork

def preprocess(image_path):
    image = Image.open(image_path) #.convert('L')  # Convert to grayscale
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    #image_array = image_array.reshape(256 * 256 * 3)
    return image_array


# Define directories and types
data_dir = "datasets/nikeTrain"
types = ["Nike", "Adidas", "Puma"]

# Initialize and train the model
model = NikeNeuralNetwork()


stop_at = 100

num_classes = len(types)
for i in range(len(types)):
    count = 0
    category_dir = os.path.join(data_dir, types[i])
    print(f"Checking directory: {category_dir}")
    if not os.path.exists(category_dir):
        print(f"Directory {category_dir} does not exist.")
        continue
    for filename in os.listdir(category_dir):
        file_path = os.path.join(category_dir, filename)

        print(file_path)
        if not os.path.isfile(file_path):
            continue
        
        img = preprocess(file_path)
        if len(img.shape) < 3:
            continue
        if img.shape[2] != 3:
            continue

        img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 256, 256)
        label = i
        target = np.zeros(num_classes)
        target[label] = 1.0

        model.train(img, target)

        count += 1
        if count == stop_at:
            break

       


# Save the trained model
torch.save(model.state_dict(), "Nike.pth")
