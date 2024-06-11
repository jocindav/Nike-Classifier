import torch
import torch.nn as nn

class NikeNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__() 
        self.model = nn.Sequential(
            nn.Linear(128 * 128, 200),  # Adjusted input size for grayscale images #layer 1
            nn.Sigmoid(),
            nn.Linear(200, 3),  # Adjusted output size for 3 classes #layer 2
            nn.Sigmoid(),
        )
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) #trying to remove the loss

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        return self.model(inputs)
    
    def train_model(self, inputs, targets, epochs=3):
        targets = torch.FloatTensor(targets)
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            loss = self.loss_function(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def train_step(self, inputs, targets):
        targets = torch.FloatTensor(targets)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
