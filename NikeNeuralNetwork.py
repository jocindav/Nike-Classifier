import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,
    
    def forward(self, x):
        return x.view(*self.shape)

class NikeNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__() 
        """
        self.model = nn.Sequential(
            nn.Linear(256 * 256 * 3, 4000),  # Adjusted input size for grayscale images #layer 1
            nn.ReLU(),
            
            nn.Linear(4000, 1000),  # Adjusted output size for 3 classes #layer 2
            nn.ReLU(),

            nn.Linear(1000, 500),  # Adjusted output size for 3 classes #layer 2
            nn.ReLU(),

            nn.Linear(500, 3),  # Adjusted output size for 3 classes #layer 2
            nn.Sigmoid(),
        )
        """

        self.model = nn.Sequential(
             nn.Conv2d(3, 256, kernel_size=8, stride=2),
             nn.BatchNorm2d(256),
             nn.ReLU(),

             nn.Conv2d(256, 256, kernel_size=8, stride=2),
             nn.BatchNorm2d(256),
             nn.ReLU(),

            nn.Conv2d(256, 3, kernel_size=8, stride=2),
            nn.ReLU(),

            View(2028),
            nn.Linear(2028, 3),
            nn.Sigmoid()

         )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) #trying to remove the loss

    def forward(self, inputs):
        #inputs = torch.FloatTensor(inputs)
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

    def train(self, inputs, targets):
        targets = torch.FloatTensor(targets)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
