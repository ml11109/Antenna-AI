import torch
from torch.utils.data import Dataset, DataLoader
import math


class TestDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = TestDataset(
    x_data=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
    y_data=torch.tensor([[1], [2], [3], [4], [5], [6]])
)

n_epochs = 10
batch_size = 2
n_samples = len(dataset)
n_iterations = math.ceil(n_samples / batch_size)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    for i, (x_batch, y_batch) in enumerate(dataloader):
        print(f"Epoch: {epoch}, Iteration: {i}, x: {x_batch}, y: {y_batch}")
