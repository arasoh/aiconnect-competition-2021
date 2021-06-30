import aiconnect.validation as val

import numpy as np

from sklearn.svm import LinearSVC, SVC


class Linear:
    def __init__(self, random_state: int = 0, tol=1e-5) -> None:
        self.random_state = random_state
        self.tol = tol

    def forward(self) -> None:
        clf = LinearSVC(random_state=self.random_state, tol=self.tol)

        return clf


class NonLinear:
    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def forward(self):
        clf = SVC(C=self.C)

        return clf


class SVM:
    def __init__(self, target: str = "linear", params: dict = None):
        if target is "lin":
            if params is not None:
                clf = Linear(
                    random_state=params["random_state"],
                    tol=params["tol"],
                )
            else:
                clf = Linear()

        if target is "nlin":
            if params is not None:
                clf = NonLinear(
                    C=params["C"],
                )
            else:
                clf = NonLinear()

        self.model = clf.forward()

    def model_training(self, data, label):
        self.model.fit(data, label)

    def label_prediction(self, data):
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
