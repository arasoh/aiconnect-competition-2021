import numpy as np
from sklearn.preprocessing import LabelEncoder


class Encoder:
    def __init__(self):
        pass

    def encode_labels(self, labels):
        le = LabelEncoder()
        le.fit(labels)
        encoded_labels = le.transform(labels)

        return encoded_labels


class Decoder:
    def __init__(self) -> None:
        pass

    def user_appearances(self, dataframe) -> dict:
        user_count = 0
        data_counts = []
        user_emails = []

        for data in dataframe.itertuples():
            email = data.EMAIL

            if email not in user_emails:
                user_emails.append(email)
                user_count += 1
                data_counts.append(1)
            else:
                data_counts[-1] += 1

        user_data = zip(user_emails, data_counts)

        for iter, (user_email, data_count) in enumerate(user_data):
            if iter <= 0:
                users = {user_email: data_count}
            else:
                users[user_email] = data_count

        return users

    def squeeze_predictions(self, pred, appearances: dict, margin: float = 0.4) -> None:
        slice_start = 0
        slice_count = 0

        cn_max = 1 - margin
        mci_max = 2 - margin

        pred = np.squeeze(pred)
        new_pred = []

        for key, value in appearances.items():
            slice_count = slice_count + value
            scores = pred[slice_start:slice_count]

            average_score = np.average(scores)

            if average_score <= cn_max:
                final_score = 0
            elif (cn_max < average_score) and (average_score <= mci_max):
                final_score = 1
            else:
                final_score = 2

            new_pred.append(final_score)
            slice_start = slice_count

        new_pred = np.array(new_pred).reshape(-1, 1)

        return new_pred

    def decode_labels(self, pred, diagnosis: dict):
        diagnosis_names = list(diagnosis.keys())
        new_pred = []

        for pred_code in np.squeeze(pred):
            new_pred.append(diagnosis_names[pred_code])

        return new_pred
