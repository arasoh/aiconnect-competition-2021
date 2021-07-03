import numpy as np

from sklearn.preprocessing import MinMaxScaler


class Normalizer:
    def __init__(self) -> None:
        pass

    def normalize(self, array):
        scaler = MinMaxScaler()
        scaler.fit(array)
        normalized_array = scaler.transform(array)

        return normalized_array
