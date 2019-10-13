from unittest import TestCase
from src.main.naive_bayes_classifier.naive_bayes_classifier import *

class TestNaivesBayesClassifier(TestCase):

    def test_extract_tokenized_vocabulary(self):
        data = np.array([['my name, isn\'t Joe!', 'name'], ['what\'s his: name?', 'name'], ['I; can\'t believe. it.', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        expected = ['my', 'name', ',', 'is', 'n\'t', 'Joe', '!', 'what', '\'s', 'his', ':', 'name', '?', 'I', ';', 'ca', 'n\'t', 'believe', '.', 'it', '.']
        actual = classifier.extract_vocabulary(data[:, :-1])
        self.assertEqual(expected, actual)

    def test_clean_unfiltered_data(self):
        classifier = NaiveBayesClassifier(np.array([['hello', 'greetings']]))
        words = ['he.', 'qa!', 'ggd;', 'ds:', 'df?', 'djdk!']
        expected = ['he', '.', 'qa', '!', 'ggd', ';', 'ds', ':', 'df', '?', 'djdk', '!']
        actual = classifier.clean_unfiltered_punctuation(words)
        self.assertEqual(expected, actual)

    def test_extract_stemmed_vocabulary(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        expected = ['foot', 'cat', 'wolf', 'talked', 'caress']
        actual = classifier.extract_vocabulary(data[:, :-1])
        self.assertEqual(expected, actual)

    def test_get_docs_by_label(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        expected = (['caresses', 'feet cats', 'wolves talked'], [1, 2, 2])
        actual = classifier.get_docs_by_label()
        self.assertEqual(expected, actual)

    def test_get_priors(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        docs = classifier.get_docs_by_label()
        expected = ([1 / 5, 2 / 5, 2 / 5])
        actual = classifier.get_priors(docs[1], np.array(docs[1]).sum())
        self.assertEqual(expected, actual)

