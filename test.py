import os
import numpy as np
import torch
from NikeNeuralNetwork import NikeNeuralNetwork
from PIL import Image

def preprocess(image_path):
    image = Image.open(image_path) #.convert('L')  #  grayscale
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    #image_array = image_array.reshape(256 * 256 * 3)
    return image_array


model = NikeNeuralNetwork()
model.load_state_dict(torch.load("Nike.pth"))

# Define directories and types
data_dir = "datasets/nikeTest"
types = ["Nike", "Adidas", "Puma"]

stop_at = 30

correct = 0
total = 0

things = [0, 0, 0]

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

        total += 1
        output = model.forward(img).detach().numpy()
        print(output)
        guess = np.argmax(output)
        if guess == label:
            correct += 1
            things[label] += 1

        count += 1
        if count == stop_at:
            break

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
print(things)
