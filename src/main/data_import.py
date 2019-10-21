import numpy as np
import os


def get_labels(train_labels):
    return np.sort(np.unique(train_labels))


def convert_to_array(data_tuple):
    return np.array(data_tuple)


def get_clean_data_set_as_array(train_data):
    train = convert_to_array(train_data)
    return np.transpose(train)


class DataImport:

    path = os.path.dirname(os.path.abspath(__file__))
    data_train = os.path.join(path, "resources/data_train.pkl")
    data_test = os.path.join(path, "resources/data_test.pkl")

    def get_train_data_as_tuple(self):
        return np.load(self.data_train, allow_pickle=True)

    def get_test_data_as_list(self):
        return np.load(self.data_test, allow_pickle=True)

