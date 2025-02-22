import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
INPUT_SIZE = 28  # 1 row at a time
SEQUENCE_LENGTH = 28  # 28 features in each row
HIDDEN_SIZE = 256
LAYERS = 2
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(self.hidden_size * SEQUENCE_LENGTH, NUM_CLASSES)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out: torch.Tensor = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# load data
train_data = MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, BATCH_SIZE, True)
test_data = MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, BATCH_SIZE, False)


# initialise network
model = RNN(INPUT_SIZE, HIDDEN_SIZE, LAYERS, NUM_CLASSES).to(device)


# loss and optimiser
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# train
for epoch in range(EPOCHS):
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data: torch.Tensor = data.to(device).squeeze(1)
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
            x = x.to(device).squeeze(1)
            y = y.to(device)
            scores = model(x)  # 64x10
            _, predictions = torch.max(scores, dim=1)  # returns -> values, indices

            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    accuracy = (num_correct / num_samples) * 100
    return accuracy


acc = check_accuracy(test_loader, model)
print(acc)  # 97.91
