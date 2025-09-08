import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class TestDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = x_data.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return torch.tensor(x), torch.tensor(y)


class MulXTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        return x * self.factor, y


class AddXTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        return x + self.factor, y


dataset = TestDataset(
    x_data=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]),
    y_data=np.array([[1], [2], [3], [4], [5], [6], [7]]),
    transform=torchvision.transforms.Compose([ToTensor(), MulXTransform(factor=2), AddXTransform(factor=10)])
)

n_epochs = 10
batch_size = 2
n_samples = len(dataset)
n_iterations = math.ceil(n_samples / batch_size)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    for i, (x_batch, y_batch) in enumerate(dataloader):
        print(f"Epoch: {epoch}, Iteration: {i}, x: {x_batch}, y: {y_batch}")
