import aiconnect.validation as val

import numpy as np

from sklearn.svm import LinearSVC, SVC


class Classifier:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma="scale",
        prob: bool = False,
        tol: float = 1e-3,
        verbose: bool = False,
        state: int = None,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.prob = prob
        self.tol = tol
        self.verbose = verbose
        self.state = state

    def forward(self):
        clf = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            probability=self.prob,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.state,
        )

        return clf


class SVM:
    def __init__(self, params: dict = None):

        if params is not None:
            clf = Classifier(
                C=params["C"],
                kernel=params["kernel"],
                degree=params["degree"],
                gamma=params["gamma"],
                prob=params["prob"],
                tol=params["tol"],
                verbose=params["verbose"],
                state=params["state"],
            )
        else:
            clf = Classifier()

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
