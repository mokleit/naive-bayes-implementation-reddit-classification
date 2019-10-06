from unittest import TestCase
from src.main.random_classifier import *
from src.main.data_import import *
import numpy as np

class TestRandomClassifier(TestCase):

    def save_predictions(self):
        data_import = DataImport()
        train_data = data_import.get_train_data_as_tuple()
        clean_train_data = data_import.get_clean_data_set_as_array(train_data)
        test_data = data_import.get_test_data_as_list()
        predictions = RandomClassifier(clean_train_data).predict(test_data)
        indexes = np.arange(0, len(predictions))
        preds = np.transpose(np.array([indexes, predictions]))
        print(preds)
        np.savetxt("random_classifier_predictions.csv", preds, fmt="%s")

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
        error_rate = classifier.get_error(test_data)
        self.assertIs(type(error_rate), np.float64)
