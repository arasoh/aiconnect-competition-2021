import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils


class NeuralNetwork(nn.modules):
    def __init__(self):
        super(self).__init__()
        self.dropout_rate = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(63, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(4 * 4 * 128, 256, bias=True)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.dropout_rate),
        )
        self.fc2 = nn.Linear(256, 3, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)

        return out


def neural_network(dataset, labels):
    device = "cuda" if cuda.is_available() else "cpu"

    # for reproducibility
    torch.manual_seed(777)
    if device is "cuda":
        cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epochs = 16
    batch_size = 128

    dataset = torch.FloatTensor(dataset)
    labels = torch.LongTensor(labels)

    data_loader = utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = NeuralNetwork().to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_batch = len(data_loader)
    model.train()

    print("Training begins...")
    for epoch in range(training_epochs):
        average_cost = 0

        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            hypothesis = model(data)
            cost = loss(hypothesis, label)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            average_cost += cost / total_batch

    print("Training is completed.")
