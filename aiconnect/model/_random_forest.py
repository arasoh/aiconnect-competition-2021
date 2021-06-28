from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, depth: int = 10, state: int = 5) -> None:
        self.classifier = RandomForestClassifier(max_depth=depth, random_state=state)

    def model_training(self, train_dataset, train_labels):
        self.classifier.fit(train_dataset, train_labels)

    def label_prediction(self, test_dataset):
        pred = self.classifier.predict(test_dataset)

        return pred
