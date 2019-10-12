from unittest import TestCase
from src.main.random_classifier.random_classifier import *
import numpy as np

class TestRandomClassifier(TestCase):

    min_error_rate = 1.

    def test_save_predictions(self):
        data_import = DataImport()
        clean_train_data = data_import.get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        classifier = RandomClassifier(clean_train_data)
        predictions = classifier.predict_weighted(data_import.get_test_data_as_list())
        classifier.save_predictions(predictions)


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

        for j in range(1000):
            predictions = classifier.predict(test_data[:, :-1])
            error_rate = classifier.get_error(predictions, test_data[:, -1])
            if error_rate < self.min_error_rate:
                self.min_error_rate = error_rate
        self.assertIs(type(error_rate), np.float64)
