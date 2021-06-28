import aiconnect.model as model
import aiconnect.preprocessing as preprocessing
import aiconnect.validation as metrics

import numpy as np


def app():
    train_path = "./data/train"
    test_path = "./data/test"
    diagnosis = {"CN": 0, "MCI": 1, "Dem": 2}

    df = preprocessing.Dataframe()

    train_data, train_labels = df.build_dataframe(train_path, train=True)
    test_data, test_labels = df.build_dataframe(test_path, train=False)

    drop_columns = [
        "summary_date",
        "activity_class_5min",
        "activity_met_1min",
        "sleep_hr_5min",
        "sleep_hypnogram_5min",
        "sleep_rmssd_5min",
        "timezone",
        "sleep_total",
    ]
    train_data = df.drop_columns(train_data, columns=drop_columns)
    test_data = df.drop_columns(test_data, columns=drop_columns)
    del drop_columns

    train_labels = df.encode_labels(train_labels, diagnosis)
    train_dataset = train_labels.join(train_data.set_index("EMAIL"), on="user_email")

    drop_columns = [
        "diagnosis_name",
        "user_email",
    ]
    train_dataset = df.drop_columns(train_dataset, columns=drop_columns)
    del drop_columns

    drop_columns = [
        "EMAIL",
    ]
    test_dataset = df.drop_columns(test_data, columns=drop_columns)
    del drop_columns

    # TODO: Convert 부분 활용방안 구상이 필요함, 일단은 인덱스 하드코딩해서 제외함
    train_dataset = train_dataset.iloc[:, :52].to_numpy(dtype=float)
    test_dataset = test_dataset.iloc[:, :51].to_numpy(dtype=float)

    train_data_array = train_dataset[:, 1:]
    train_label_array = train_dataset[:, 0]

    norm = preprocessing.Normalizer()
    enc = preprocessing.Encoder()

    train_data = norm.normalize(train_data_array)
    test_data = norm.normalize(test_dataset)

    train_labels = enc.encode_labels(train_label_array)
    train_labels = train_labels.reshape(train_labels.size, 1)

    # users = df.encode_users(train_data)

    randf = model.RandomForest()
    randf.model_training(train_data, train_labels)
    pred = randf.label_prediction(test_dataset)

    print("breakpoint")
    # nn = model.NeuralNetwork()
    #
    # metric = metrics.Metrics

    return 0


if __name__ == "__main__":
    app()
