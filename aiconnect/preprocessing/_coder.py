from sklearn.preprocessing import LabelEncoder


class Encoder:
    def __init__(self):
        pass

    def encode_labels(self, labels):
        le = LabelEncoder()
        le.fit(labels)
        encoded_labels = le.transform(labels)

        return encoded_labels
