'''
Run on Google Colab and not locally. For local machine, it might give error in dowloading the datasets from server.
'''

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create model
class NeuralNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.layer0 = nn.Linear(input_size, 50)  # input_layer
        self.layer1 = nn.Linear(50, num_classes)  # fully connected layer 1

    def forward(self, x):
        # x -> input_layer -> a0 -> Relu -> a1 -> fc_layer1 -> a2 ->
        a0 = self.layer0(x)
        a1 = F.relu(a0)
        a2 = self.layer1(a1)
        return a2


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5


# load data
train_data = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, BATCH_SIZE, True)
test_data = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, BATCH_SIZE, False)


# initialise network
model = NeuralNet(INPUT_SIZE, NUM_CLASSES).to(device)


# loss and optimiser
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# train
for epoch in range(EPOCHS):
    for batch_index, (data, targets) in enumerate(train_loader):
        data: torch.Tensor = data.to(device)
        targets = targets.to(device)

        data = data.view(data.shape[0], -1)

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
            x = x.view(x.shape[0], -1)
            y = y.to(device)
            scores = model(x)  # 64x10
            _, predictions = torch.max(scores, dim=1)  # returns -> values, indices

            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    accuracy = (num_correct / num_samples) * 100
    return accuracy


acc = check_accuracy(test_loader, model)
print(acc)  # 96.53
