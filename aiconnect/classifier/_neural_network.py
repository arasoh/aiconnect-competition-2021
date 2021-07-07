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


class NeuralNetwork:
    def __init__(self, n_features: int, path: str) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.n_features = n_features
        self.path = path

    def model_training(self, data, labels):
        # for reproducibility
        torch.manual_seed(1)
        if self.device is "cuda":
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

        model = _4LayerPerceptron(n_features=self.n_features).to(self.device)

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        total_batch = len(data_loader)

        print("Training begins...")

        for epoch in range(training_epochs):
            average_cost = 0

            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                hypothesis = model(data)
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
        torch.save(model.state_dict(), NN_PATH)

        return 0

    def label_prediction(self, data):
        data = torch.FloatTensor(data).to(self.device)

        model = _4LayerPerceptron(n_features=self.n_features).to(self.device)
        model.load_state_dict(torch.load(self.path))

        pred = model(data)
        _, pred = torch.max(pred, 1)

        pred = pred.to("cpu").numpy()

        return pred

    def f1_score(self, true, pred):
        cn_index = 0
        mci_index = 0
        dem_index = 0

        metrics = val.Metrics(true, pred)

        """
            CN score
            """
        cn_precision = metrics.precision(index=cn_index)
        cn_recall = metrics.recall(index=cn_index)
        cn_f1_score = metrics.f1_score(cn_precision, cn_recall)

        """
            MCI score
            """
        mci_precision = metrics.precision(index=mci_index)
        mci_recall = metrics.recall(index=mci_index)
        mci_f1_score = metrics.f1_score(mci_precision, mci_recall)

        """
            Dem score
            """
        dem_precision = metrics.precision(index=dem_index)
        dem_recall = metrics.recall(index=dem_index)
        dem_f1_score = metrics.f1_score(dem_precision, dem_recall)

        scores = [cn_f1_score, mci_f1_score, dem_f1_score]

        f1_score = metrics.macro_f1_score(scores)

        return f1_score
