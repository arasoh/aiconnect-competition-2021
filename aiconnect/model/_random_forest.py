import aiconnect.validation as val

from sklearn.ensemble import RandomForestClassifier


class Classifier:
    def __init__(
        self,
        n_estimators: int = 100,
        depth: int = 10,
        state: int = 5,
        verbose: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.depth = depth
        self.state = state
        self.verbose = verbose

    def forward(self):
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.depth,
            random_state=self.state,
            verbose=self.verbose,
        )

        return clf


class RandomForest:
    def __init__(self, params: dict = None) -> None:
        if params is not None:
            clf = Classifier(
                n_estimators=params["n_estimators"],
                depth=params["depth"],
                state=params["state"],
                verbose=params["verbose"],
            )
        else:
            clf = Classifier()
        self.model = clf.forward()

    def model_training(self, data, labels) -> None:
        self.model.fit(data, labels)

    def label_prediction(self, data) -> None:
        pred = self.model.predict(data)

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
