import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5
NUM_FEATURES = 784
NUM_CLASSES = 10
HIDDEN_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        return out

model = NeuralNet(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

with torch.no_grad():
    num_correct, num_samples = 0, 0
    for images, labels in test_loader:
        num_samples += images.shape[0]
        num_correct += model(images.reshape(-1, 28*28)).eq(labels).sum().item()

    print(f"Accuracy: {num_correct / num_samples * 100:.2f}%")
