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

    def decode_slash(self, dataframe):
        decoded_features = []

        for data in dataframe.itertuples():
            features = [
                data[1],
                data[2],
                data[3],
                data[4],
                data[5],
            ]

            decoded_feature = []

            stop_words = ["", "'", "[", "]"]

            for _, feature in enumerate(features):
                feature_temp = feature.split("/")

                if len(feature_temp) > 1:
                    feature = feature_temp
                else:
                    feature_temp = feature.lstrip("[").rstrip("]")
                    feature = feature_temp.split(",")

                try:
                    for stop_word in stop_words:
                        feature.remove(stop_word)
                except:
                    pass

                feature = np.array(feature, dtype=np.float)

                feature_avg = np.average(feature)
                feature_std = np.std(feature)

                decoded_feature.extend(
                    [
                        feature_avg,
                        feature_std,
                    ]
                )

            decoded_features.append(decoded_feature)

        decoded_features = np.array(decoded_features)

        return decoded_features

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

        cn_max = 1 - (margin**2)
        mci_max = 2 - (margin**2)

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
