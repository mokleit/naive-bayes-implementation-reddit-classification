import numpy as np
import os

class DataImport:

    path = os.path.dirname(os.path.abspath(__file__))
    data_train = os.path.join(path, "resources/data_train.pkl")
    data_test = os.path.join(path, "resources/data_test.pkl")

    def get_train_data(self):
        return np.load(self.data_train, allow_pickle=True)

    def get_test_data(self):
        return np.load(self.data_test, allow_pickle=True)

    def convert_to_array(self, tuple):
        return np.array(tuple)
