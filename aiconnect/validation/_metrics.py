import numpy as np

from sklearn.metrics import confusion_matrix


class Metrics:
    def __init__(self, true, pred) -> None:
        self.confusion_matrix = confusion_matrix(true, pred)

    def precision(self, index: int) -> float:
        precision_array = self.confusion_matrix[:, index]

        true_positive = precision_array[index]
        total_positive = np.sum(precision_array, dtype=np.int32)

        precision = true_positive / total_positive

        return precision

    def recall(self, index: int) -> float:
        recall_array = self.confusion_matrix[index, :]

        true_positive = recall_array[index]
        total_true = np.sum(recall_array, dtype=np.int32)

        recall = true_positive / total_true

        return recall

    def f1_score(self, precision: float, recall: float) -> float:
        score = 2 * (precision * recall) / (precision + recall)

        return score

    def macro_f1_score(self):
        cn_index = 0
        mci_index = 1
        dem_index = 2

        """Cognitive normal(CN) score"""
        cn_precision = self.precision(index=cn_index)
        cn_recall = self.recall(index=cn_index)
        cn_f1_score = self.f1_score(cn_precision, cn_recall)

        """Mild cognitive impairment(MCI) score"""
        mci_precision = self.precision(index=mci_index)
        mci_recall = self.recall(index=mci_index)
        mci_f1_score = self.f1_score(mci_precision, mci_recall)

        """Dementia(Dem) score"""
        dem_precision = self.precision(index=dem_index)
        dem_recall = self.recall(index=dem_index)
        dem_f1_score = self.f1_score(dem_precision, dem_recall)

        f1_scores = [cn_f1_score, mci_f1_score, dem_f1_score]

        macro_f1_score = round(sum(f1_scores) / 3, 2)

        return macro_f1_score
