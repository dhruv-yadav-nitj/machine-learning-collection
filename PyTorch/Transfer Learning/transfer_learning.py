import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import (Dataset, DataLoader)
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import os
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NUM_CLASSES = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EPOCHS = 5


class FineTunedVGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.vgg16(weights='DEFAULT')

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.avgpool = nn.Identity()  # nn.Identity() is just a placeholder -> kinda buffer -> output = input
        self.base_model.classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        pred = self.base_model(x)
        return pred


model = FineTunedVGG(NUM_CLASSES).to(device)

# DataLoading
dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# Loss_Fn and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)  # only updates those layers' parameters which we are training

model.train()

for epoch in range(EPOCHS):
    losses = []
    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)

        loss = criterion(scores, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print(f'Epoch: {epoch}  Avg. Loss:  {round(sum(losses)/len(losses), 2)}')


def check_accuracy(loader, model):
    num_correct = num_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_total += predictions.size(0)

        acc = num_correct/num_total
        return acc


accuracy = check_accuracy(test_loader, model)
print(accuracy)  # 61
