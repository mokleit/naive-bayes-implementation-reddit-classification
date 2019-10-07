from unittest import TestCase
from src.main.random_classifier import *
import numpy as np

class TestRandomClassifier(TestCase):

    def test_save_predictions(self):
        data_import = DataImport()
        clean_train_data = data_import.get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        RandomClassifier(clean_train_data).save_predictions(data_import.get_test_data_as_list())

    def test_predictions(self):
        train_data = np.ones((10, 2))
        test_data = np.ones((10, 2))
        labels = [10, 20, 30]
        for j in range(len(train_data)):
            random.seed(datetime.now())
            train_data[j, -1] = random.choice(labels)
            random.seed(datetime.now())
            test_data[j, -1] = random.choice(labels)

        actual = RandomClassifier(train_data).predict(test_data)
        self.assertEqual(len(actual), len(test_data))


    def test_error(self):
        train_data = np.ones((10, 2))
        test_data = np.ones((10, 2))
        labels = [10, 20, 30]
        for j in range(len(train_data)):
            random.seed(datetime.now())
            train_data[j, -1] = random.choice(labels)
            random.seed(datetime.now())
            test_data[j, -1] = random.choice(labels)

        classifier = RandomClassifier(train_data)

        error_rate = []
        for j in range(100):
            error_rate.append(classifier.get_error(test_data))

        average_error_rate = np.mean(error_rate)
        self.assertIs(type(average_error_rate), np.float64)
