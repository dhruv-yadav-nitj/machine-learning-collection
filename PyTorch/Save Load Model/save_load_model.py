import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

'''
Documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
'''

FILE_NAME = 'model.pth.tar'


def save_model(state, filename=FILE_NAME):
    torch.save(state, filename)


def load_model(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    model = torchvision.models.vgg16(weights=None)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    checkpoint = {'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}

    save_model(checkpoint)

    checkpoint = torch.load(FILE_NAME)

    load_model(checkpoint, model, optimizer)
