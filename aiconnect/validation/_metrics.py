import numpy as np


class Metrics:
    def __init__(self) -> None:
        pass

    def precision(self, confusion_matrix, index: int) -> float:
        precision_array = confusion_matrix[:, index]

        true_positive = precision_array[index]
        total_positive = np.sum(precision_array, dtype=np.int32)

        precision = true_positive / total_positive

        return precision

    def recall(self, confusion_matrix, index: int) -> float:
        recall_array = confusion_matrix[index, :]

        true_positive = recall_array[index]
        total_true = np.sum(recall_array, dtype=np.int32)

        recall = true_positive / total_true

        return recall

    def f1_score(self, precision: float, recall: float) -> float:
        score = 2 * (precision * recall) / (precision + recall)

        return score

    def macro_f1_score(self, f1_scores: list) -> float:
        macro_score = round(sum(f1_scores) / 3, 2)

        return macro_score
