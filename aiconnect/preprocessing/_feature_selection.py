from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif


class Selector:
    def __init__(self, k: int = 10):
        self.k = k

    def forward(self):
        slt = SelectKBest(chi2, k=self.k)

        return slt


class KBest:
    def __init__(self, params: dict = None):
        if params is not None:
            slt = Selector(
                k=params["k"],
            )
        else:
            slt = Selector()

        self.model = slt.forward()

    def model_training(self, data, labels):
        self.model.fit(data, labels)

    def feature_indices(self):
        feature_indices = self.model.get_support(indices=True)

        return feature_indices
