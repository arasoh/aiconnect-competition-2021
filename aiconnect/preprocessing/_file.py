import csv
import numpy as np
import pandas as pd


class File:
    def __init__(self):
        pass

    def build_dataframe(self, csv_path, train=True):
        if train is True:
            data_path = csv_path + "/train.csv"
            label_path = csv_path + "/train_label.csv"
        else:
            data_path = csv_path + "/test.csv"
            label_path = None

        dataframe = pd.read_csv(data_path)

        if label_path is not None:
            labels = pd.read_csv(label_path)
        else:
            labels = None

        return dataframe, labels

    def select_columns(self, dataframe, columns: list) -> None:
        dataframe = dataframe.filter(items=columns)

        return dataframe

    def drop_columns(self, dataframe, columns: list) -> None:
        dataframe = dataframe.drop(columns, axis=1)

        return dataframe

    def encode_users(self, dataframe):
        user_count = 0
        user_id = []
        user_email = []

        for data in dataframe.itertuples():
            if data.EMAIL not in user_email:
                user_id.append(user_count)
                user_email.append(data.EMAIL)
                user_count += 1
            else:
                pass

        user_array = np.array([user_id, user_email], ndmin=2).transpose()
        user_dataframe = pd.DataFrame(user_array, columns=["user_index", "user_email"])

        return user_dataframe

    # def encode_true(self, dataframe, diagnosis: dict):
    #     for data in dataframe.itertuples():
    #         if data.Index is 0:
    #             array = np.array(
    #                 [
    #                     diagnosis[data.DIAG_NM],
    #                 ],
    #                 ndmin=2,
    #             )
    #         else:
    #             array_temp = np.array(
    #                 [
    #                     diagnosis[data.DIAG_NM],
    #                 ],
    #                 ndmin=2,
    #             )
    #             array = np.concatenate((array, array_temp), axis=0)
    #
    #         new_dataframe = pd.DataFrame(
    #             array,
    #             columns=["diagnosis_index", "diagnosis_name", "user_email"],
    #         )
    #
    #     return new_dataframe

    def encode_labels(self, dataframe, diagnosis: dict):
        for data in dataframe.itertuples():
            if data.Index is 0:
                data_array = np.array(
                    [
                        diagnosis[data.DIAG_NM],
                        data.DIAG_NM,
                        data.SAMPLE_EMAIL,
                    ],
                    ndmin=2,
                )
            else:
                data_array_temp = np.array(
                    [
                        diagnosis[data.DIAG_NM],
                        data.DIAG_NM,
                        data.SAMPLE_EMAIL,
                    ],
                    ndmin=2,
                )
                data_array = np.concatenate((data_array, data_array_temp), axis=0)

        new_dataframe = pd.DataFrame(
            data_array,
            columns=["diagnosis_code", "diagnosis_name", "user_email"],
        )

        return new_dataframe

    def write_csv(self, data, path):
        csv_path = path + "/submission.csv"
        data = np.array(data, ndmin=2).transpose()

        dataframe = pd.DataFrame(data, columns=["ID", "DIAG_NM"])

        dataframe.to_csv(csv_path)

        return "CSV 파일이 생성됨"
