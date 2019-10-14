import unittest
from src.main.smooth_naive_bayes_classifier.smooth_naive_bayes_classifier import *
from src.main.data_import import *

class TestSmoothNaiveBayes(unittest.TestCase):

    def test_posteriors(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = SmoothNaiveBayesClassifier(data)
        expected = [[1/6, 1/6,1/6,1/6,2/6], [2/7, 2/7, 1/6, 1/6,1/6], [1/6,1/6,2/7,2/7,1/6]]
        classifier.train()
        self.assertEqual(expected, classifier.posteriors)

    def test_predict_with_unexisting_words(self):
        data = np.array([['my feet arms head neck', 'body'], ['cats and dogs are animals', 'animal'], ['wolves and bears are dangerous', 'animal'], ['your mouth his ankle', 'body'], ['obama macron trudeau liberal democrats', 'politics']])
        classifier = SmoothNaiveBayesClassifier(data)
        test_data = [['my head hurts and my neck is soar'], ['dogs against cats and bears against wolves'], ['macron is friends with obama and trudeau but remember this is politics']]
        expected = ['body', 'animal', 'politics']
        classifier.train()
        actual = classifier.predict(test_data)
        self.assertEqual(expected, actual)

    @unittest.skip('enable only for saving predictions.')
    def test_save_predictions(self):
        data_import = DataImport()
        clean_train_data = data_import.get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        classifier = SmoothNaiveBayesClassifier(clean_train_data)
        classifier.train()
        predictions = classifier.predict(data_import.get_test_data_as_list())
        classifier.save_predictions("smooth_naive_bayes_classifier_predictions", predictions)
