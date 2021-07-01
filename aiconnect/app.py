import aiconnect.model as model
import aiconnect.preprocessing as prep

import numpy as np


def app():
    train_path = "./data/train"
    test_path = "./data/test"
    diagnosis = {"CN": 0, "MCI": 1, "Dem": 2}

    f = prep.File()
    enc = prep.Encoder()

    train_data, train_labels = f.build_dataframe(train_path, train=True)
    test_data, _ = f.build_dataframe(test_path, train=False)

    train_identifiers = f.select_columns(train_data, ["EMAIL"])
    test_identifiers = f.select_columns(test_data, ["EMAIL"])

    train_labels = f.encode_labels(train_labels, diagnosis)
    train_true = f.select_columns(train_labels, ["diagnosis_code"])
    train_true = enc.encode_labels(train_true.to_numpy()).reshape(-1, 1)

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
    train_data = f.drop_columns(train_data, columns=drop_columns)
    test_data = f.drop_columns(test_data, columns=drop_columns)
    del drop_columns

    train_dataset = train_labels.join(train_data.set_index("EMAIL"), on="user_email")

    drop_columns = [
        "diagnosis_name",
        "user_email",
    ]
    train_dataset = f.drop_columns(train_dataset, columns=drop_columns)
    del drop_columns

    drop_columns = [
        "EMAIL",
    ]
    test_dataset = f.drop_columns(test_data, columns=drop_columns)
    del drop_columns

    # TODO: Convert 부분 활용방안 구상이 필요함, 일단은 인덱스 하드코딩해서 제외함
    train_dataset = train_dataset.iloc[:, :52].to_numpy(dtype=float)
    test_dataset = test_dataset.iloc[:, :51].to_numpy(dtype=float)

    train_data_array = train_dataset[:, 1:]
    train_label_array = train_dataset[:, 0]

    norm = prep.Normalizer()

    # TODO: CSV 파일 내보내기 기능 구현 및 함수 테스트 중

    train_data = norm.normalize(train_data_array)
    test_data = norm.normalize(test_dataset)

    train_labels = enc.encode_labels(train_label_array)
    train_labels = train_labels.reshape(train_labels.size, 1)

    # users = df.encode_users(train_data)

    dec = prep.Decoder()

    train_appearances = dec.user_appearances(train_identifiers)
    test_appearances = dec.user_appearances(test_identifiers)

    """
    Classifcation Models
    """

    ### Support Vector Machine
    # train_params = {
    #     "C": 10,
    #     "kernel": "rbf",
    #     "degree": 3,
    #     "prob": True,
    #     "tol": 1e-4,
    #     "verbose": False,
    #     "state": 0,
    # }
    # svm = model.SVM(params=train_params)
    #
    # train_labels = np.squeeze(train_labels)
    # svm.model_training(train_data, train_labels)
    #
    # train_pred = svm.label_prediction(train_data)
    # train_pred = dec.squeeze_predictions(train_pred, train_appearances, margin=0.4)
    #
    # train_score = svm.f1_score(train_labels, train_pred)
    #
    # test_pred = svm.label_prediction(test_dataset)

    ### Random Forest
    train_params = {
        "n_estimators": 128,
        "depth": 32,
        "state": 48,
        "verbose": True,
    }
    randf = model.RandomForest(params=train_params)

    train_labels = np.squeeze(train_labels)
    randf.model_training(train_data, train_labels)

    train_pred = randf.label_prediction(train_data)
    train_pred = dec.squeeze_predictions(train_pred, train_appearances, margin=0.4)

    test_score = randf.f1_score(train_labels, train_pred)

    test_pred = randf.label_prediction(test_dataset)

    print("breakpoint")
    # nn = model.NeuralNetwork()
    #
    # metric = metrics.Metrics

    return 0


if __name__ == "__main__":
    app()
