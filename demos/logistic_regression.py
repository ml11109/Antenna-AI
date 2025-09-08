import numpy as np
import sklearn
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


num_epochs = 1000
learning_rate = 0.03

bc_dataset = sklearn.datasets.load_breast_cancer()
X_data, y_data = bc_dataset.data, bc_dataset.target
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.2, random_state=0)

sc = sklearn.preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0], 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(y_test.shape[0], 1)

model = Model(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / y_test.shape[0]
    print(acc)
