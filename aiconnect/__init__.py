import aiconnect.classifier as clf
import aiconnect.preprocessing as prep

import numpy as np


def app():
    train_path = "./aiconnect/data/train"
    test_path = "./aiconnect/data/test"
    diagnosis = {"CN": 0, "MCI": 1, "Dem": 2}

    f = prep.File()
    enc = prep.Encoder()
    dec = prep.Decoder()

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
    train_slashes = train_dataset.iloc[:, 52:]
    test_slashes = test_dataset.iloc[:, 51:]
    decoded_train_slashes = dec.decode_slash(train_slashes)
    decoded_test_slashes = dec.decode_slash(test_slashes)

    train_dataset = train_dataset.iloc[:, :52].to_numpy(dtype=float)
    test_dataset = test_dataset.iloc[:, :51].to_numpy(dtype=float)

    train_data_array = train_dataset[:, 1:]
    train_data_array = np.concatenate((train_data_array, decoded_train_slashes), axis=1)
    train_label_array = train_dataset[:, 0]

    test_dataset = np.concatenate((test_dataset, decoded_test_slashes), axis=1)

    feature_size = test_dataset.shape[1]

    norm = prep.Normalizer()

    # TODO: CSV 파일 내보내기 기능 구현 및 함수 테스트 중

    train_data = norm.normalize(train_data_array)
    test_data = norm.normalize(test_dataset)

    train_labels = enc.encode_labels(train_label_array)
    train_labels = train_labels.reshape(train_labels.size, 1)

    # users = df.encode_users(train_data)

    train_appearances = dec.user_appearances(train_identifiers)
    test_appearances = dec.user_appearances(test_identifiers)

    """
    Classifcation Models
    """
    k_rate = 0.8
    k_params = {
        "k": int(feature_size * k_rate),
    }
    k_best = prep.KBest(k_params)
    k_best.model_training(train_data, train_labels)
    feature_indices = k_best.feature_indices()

    pred_margin = 0.6

    ### Support Vector Machine
    # train_params = {
    #     "C": 5,
    #     "kernel": "poly",
    #     "degree": 3,
    #     "gamma": "auto",
    #     "prob": True,
    #     "tol": 1e-4,
    #     "verbose": True,
    #     "state": 0,
    # }
    # svm = classifier.SVM(params=train_params)
    #
    # train_labels = np.squeeze(train_labels)
    #
    # reduced_train_data = train_data[:, feature_indices]
    #
    # svm.model_training(reduced_train_data, train_labels)
    #
    # train_pred = svm.label_prediction(reduced_train_data)
    # train_pred = dec.squeeze_predictions(
    #     train_pred,
    #     train_appearances,
    #     margin=pred_margin,
    # )
    #
    # train_score = svm.f1_score(train_true, train_pred)
    # print(train_score)
    # exit(0)
    #
    # reduced_test_data = test_data[:, feature_indices]
    #
    # test_pred = svm.label_prediction(reduced_test_data)
    # test_pred = dec.squeeze_predictions(
    #     test_pred,
    #     test_appearances,
    #     margin=pred_margin,
    # )
    #
    # test_pred_enc = dec.decode_labels(test_pred, diagnosis)
    #
    # output_data = [list(test_appearances.keys()), test_pred_enc]
    # f.write_csv(output_data, test_path)

    ### Random Forest
    # train_params = {
    #    "n_estimators": 256,
    #    "depth": 128,
    #    "split": 8,
    #    "leaf": 1,
    #    "max_features": "auto",
    #    "state": 0,
    #    "verbose": False,
    # }

    # randf = classifier.RandomForest(params=train_params)

    # train_labels = np.squeeze(train_labels)

    # reduced_train_data = train_data[:, feature_indices]

    # randf.model_training(reduced_train_data, train_labels)

    # train_pred = randf.label_prediction(reduced_train_data)
    # train_pred = dec.squeeze_predictions(
    #    train_pred,
    #    train_appearances,
    #    margin=pred_margin,
    # )

    # train_score = randf.f1_score(train_true, train_pred)
    # print(train_score)

    # reduced_test_data = test_data[:, feature_indices]
    # test_pred = randf.label_prediction(reduced_test_data)
    # test_pred = dec.squeeze_predictions(
    #    test_pred,
    #    test_appearances,
    #    margin=pred_margin,
    # )

    # test_pred_enc = dec.decode_labels(test_pred, diagnosis)

    # output_data = [list(test_appearances.keys()), test_pred_enc]
    # f.write_csv(output_data, test_path)

    ### Neural network
    NN_PATH = "./aiconnect/model/neural_network.pth"

    nn = classifier.NeuralNetwork(n_features=k_params["k"], path=NN_PATH)

    reduced_train_data = train_data[:, feature_indices]

    nn.model_training(reduced_train_data, train_labels)

    train_pred = nn.label_prediction(reduced_train_data)

    train_pred = dec.squeeze_predictions(
        train_pred,
        train_appearances,
        margin=pred_margin,
    )

    train_score = nn.f1_score(train_true, train_pred)

    print(train_score)

    return 0
