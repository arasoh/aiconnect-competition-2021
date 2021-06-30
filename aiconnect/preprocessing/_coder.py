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
