import unittest
from src.main.naive_bayes_classifier_with_validation.naive_bayes_classifier_with_validation import *
from src.main.data_import import *


class TestNaivesBayesClassifierWithValidation(unittest.TestCase):

    @unittest.skip
    def test_train(self):
        data_import = DataImport()
        data = get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        classifier = NaiveBayesClassifierWithValidation(data, 0.59)
        error_rate = classifier.train()
        self.assertLess(error_rate, 0.5)





