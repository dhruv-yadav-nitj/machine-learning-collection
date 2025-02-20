import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # (conv + max pooling) layers
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, (3,3), (1,1)),  # 8 features/filters/nodes
            nn.ReLU(),
            nn.MaxPool2d((3,3), (1,1)),
            nn.Conv2d(8, 16, (3,3), (1,1)),  # 16 feature maps to extract more features
            nn.ReLU(),
            nn.MaxPool2d((3,3), (1,1))
        )

        flattened_size = 16 * 20 * 20  # this is what we would be getting based on the above config

        self.fc1 = nn.Linear(flattened_size, self.num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x 

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
INPUT_CHANNELS = 1
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5


# load data
train_data = MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, BATCH_SIZE, True)
test_data = MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, BATCH_SIZE, False)


# initialise network
model = CNN(INPUT_CHANNELS, NUM_CLASSES).to(device)


# loss and optimiser
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# train
for epoch in range(EPOCHS):
    for batch_index, (data, targets) in enumerate(train_loader):
        data: torch.Tensor = data.to(device)
        targets = targets.to(device)

        # forward prop
        scores = model(data)
        loss = F.cross_entropy(scores, targets)

        # backward prop (autograd)
        optimizer.zero_grad()
        loss.backward()

        # gradient descent -> update model parameters
        optimizer.step()


# check accuracy
def check_accuracy(loader, model):
    num_correct = num_samples = 0
    model.eval()  # evaluation mode ON

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)  # 64x10
            _, predictions = torch.max(scores, dim=1)  # returns -> values, indices

            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    accuracy = (num_correct / num_samples) * 100
    return accuracy


acc = check_accuracy(test_loader, model)
print(acc)  # 98.58
