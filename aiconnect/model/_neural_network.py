import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils


class _4LayerPerceptron(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        dropout_rate = 0.5

        linear1 = nn.Linear(n_features, 32, bias=True)
        linear2 = nn.Linear(32, 128, bias=True)
        linear3 = nn.Linear(128, 32, bias=True)
        linear4 = nn.Linear(32, 3, bias=True)

        bn1 = nn.BatchNorm1d(32)
        bn2 = nn.BatchNorm1d(128)
        bn3 = nn.BatchNorm1d(32)

        relu = nn.ReLU()

        dropout = nn.Dropout(p=dropout_rate)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)

        self.layer1 = nn.Sequential(linear1, bn1, relu, dropout)
        self.layer2 = nn.Sequential(linear2, bn1, relu, dropout)
        self.layer3 = nn.Sequential(linear3, bn1, relu, dropout)
        self.layer4 = nn.Sequential(linear4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class NeuralNetwork:
    def __init__(self, n_features: int) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = _4LayerPerceptron(n_features=n_features).to(self.device)

    def model_training(self, data, labels):
        # for reproducibility
        torch.manual_seed(1)
        if self.device is "cuda":
            cuda.manual_seed_all(1)

        training_epochs = 16
        batch_size = 128
        learning_rate = 0.001

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)

        dataset = utils.data.TensorDataset(data, labels)
        data_loader = utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        total_batch = len(data_loader)

        print("Training begins...")
        self.model.train()
        for epoch in range(training_epochs):
            average_cost = 0

            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                hypothesis = self.model(data)
                cost = criterion(hypothesis, labels)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                average_cost += cost / total_batch

            print(
                "Epoch:",
                "%04d" % (epoch + 1),
                "cost =",
                "{:.9f}".format(average_cost),
            )

        print("Training is completed.")

    def model_validation(self, data, labels):
        pass
