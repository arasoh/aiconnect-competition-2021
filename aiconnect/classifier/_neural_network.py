import aiconnect.validation as val

import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils


class _4LayerPerceptron(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        dropout_rate = 0.25

        linear1 = nn.Linear(n_features, 128, bias=True)
        linear2 = nn.Linear(128, 512, bias=True)
        linear3 = nn.Linear(512, 128, bias=True)
        linear4 = nn.Linear(128, 3, bias=True)

        bn1 = nn.BatchNorm1d(128)
        bn2 = nn.BatchNorm1d(512)
        bn3 = nn.BatchNorm1d(128)

        relu = nn.ReLU()

        dropout = nn.Dropout(p=dropout_rate)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)

        self.layer1 = nn.Sequential(linear1, bn1, relu, dropout)
        self.layer2 = nn.Sequential(linear2, bn2, relu, dropout)
        self.layer3 = nn.Sequential(linear3, bn3, relu, dropout)
        self.layer4 = nn.Sequential(linear4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class _8LayerPerceptron(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        dropout_rate = 0.25

        linear1 = nn.Linear(n_features, 128, bias=True)
        linear2 = nn.Linear(128, 256, bias=True)
        linear3 = nn.Linear(256, 512, bias=True)
        linear4 = nn.Linear(512, 512, bias=True)
        linear5 = nn.Linear(512, 256, bias=True)
        linear6 = nn.Linear(256, 128, bias=True)
        linear7 = nn.Linear(128, 64, bias=True)
        linear8 = nn.Linear(64, 3, bias=True)

        bn1 = nn.BatchNorm1d(128)
        bn2 = nn.BatchNorm1d(256)
        bn3 = nn.BatchNorm1d(512)
        bn4 = nn.BatchNorm1d(512)
        bn5 = nn.BatchNorm1d(256)
        bn6 = nn.BatchNorm1d(128)
        bn7 = nn.BatchNorm1d(64)

        relu = nn.ReLU()

        dropout = nn.Dropout(p=dropout_rate)

        nn.init.xavier_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.xavier_uniform_(linear3.weight)
        nn.init.xavier_uniform_(linear4.weight)
        nn.init.xavier_uniform_(linear5.weight)
        nn.init.xavier_uniform_(linear6.weight)
        nn.init.xavier_uniform_(linear7.weight)
        nn.init.xavier_uniform_(linear8.weight)

        self.layer1 = nn.Sequential(linear1, bn1, relu, dropout)
        self.layer2 = nn.Sequential(linear2, bn2, relu, dropout)
        self.layer3 = nn.Sequential(linear3, bn3, relu, dropout)
        self.layer4 = nn.Sequential(linear4, bn4, relu, dropout)
        self.layer5 = nn.Sequential(linear5, bn5, relu, dropout)
        self.layer6 = nn.Sequential(linear6, bn6, relu, dropout)
        self.layer7 = nn.Sequential(linear7, bn7, relu, dropout)
        self.layer8 = nn.Sequential(linear8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


class NeuralNetwork:
    def __init__(self, dnn: str, n_features: int, path: str) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        
        self.path = path
        
        if dnn == "4-layer":
            self.model = _4LayerPerceptron(n_features=n_features).to(self.device)
        elif dnn == "8-layer":
            self.model = _8LayerPerceptron(n_features=n_features).to(self.device)

    def model_training(self, data, labels):
        # for reproducibility
        torch.manual_seed(1)
        if self.device == "cuda":
            cuda.manual_seed_all(1)

        training_epochs = 256
        batch_size = 256
        learning_rate = 0.0006

        data = torch.FloatTensor(data)
        labels = torch.LongTensor(labels)
        labels = torch.squeeze(labels)

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

        NN_PATH = self.path
        torch.save(self.model.state_dict(), NN_PATH)

        return 0

    def label_prediction(self, data):
        data = torch.FloatTensor(data).to(self.device)

        self.model.load_state_dict(torch.load(self.path))

        pred = self.model(data)
        _, pred = torch.max(pred, 1)

        pred = pred.to("cpu").numpy()

        return pred
