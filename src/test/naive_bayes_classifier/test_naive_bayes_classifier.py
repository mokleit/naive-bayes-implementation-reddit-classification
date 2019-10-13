from unittest import TestCase
from src.main.naive_bayes_classifier.naive_bayes_classifier import *

class TestNaivesBayesClassifier(TestCase):

    def test_extract_tokenized_vocabulary(self):
        data = np.array([['my name, isn\'t Joe!', 'name'], ['what\'s his: name?', 'name'], ['I; can\'t believe. it.', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        expected = ['my', 'name', ',', 'is', 'n\'t', 'Joe', '!', 'what', '\'s', 'his', ':', '?', 'I', ';', 'ca', 'believe', '.', 'it']
        actual = classifier.extract_vocabulary(data[:, :-1])
        self.assertEqual(expected, actual)

    def test_vocabulary_with_words_to_stem(self):
        data = np.array([['feet cats foot', 'name'], ['cats my wolf', 'name'], ['wolves talk', 'surprise'], ['talked caress', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        expected = ['foot', 'cat', 'my', 'wolf', 'talk', 'talked', 'caress']
        actual = classifier.extract_vocabulary(data[:, :-1])
        self.assertEqual(expected, actual)

    def test_clean_unfiltered_data(self):
        classifier = NaiveBayesClassifier(np.array([['hello', 'greetings']]))
        words = ['he.', 'qa!', 'ggd;', 'ds:', 'df?', 'djdk!']
        expected = ['he', '.', 'qa', '!', 'ggd', ';', 'ds', ':', 'df', '?', 'djdk', '!']
        actual = classifier.clean_unfiltered_punctuation(words)
        self.assertEqual(expected, actual)

    def test_get_docs_by_label(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        expected = (['caress', 'foot cat', 'wolf talked'], [1, 2, 2])
        actual = classifier.get_docs_by_label()
        self.assertEqual(expected, actual)

    def test_compute_priors(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        docs = classifier.get_docs_by_label()
        expected = ([1 / 5, 2 / 5, 2 / 5])
        classifier.compute_priors(docs[1], np.array(docs[1]).sum())
        self.assertEqual(expected, classifier.priors)

    def test_compute_posteriors(self):
        data = np.array([['feet cat', 'name'], ['cats wolf', 'name'], ['wolves', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        docs = classifier.get_docs_by_label()
        expected = [[1/3, 1/3, 1/3], [0, 0, 0]]
        classifier.compute_posteriors(docs[0][0], 0)
        self.assertEqual(expected, classifier.posteriors)

    def test_initialize_2d_list(self):
        data = np.array([['feet cat', 'name'], ['cats wolf', 'name'], ['wolves', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        expected = [[0,0,0], [0,0,0], [0,0,0]]
        actual = classifier.initialize_2d_list(3, 3, 0)
        self.assertEqual(expected, actual)

    def test_train(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        docs = classifier.get_docs_by_label()
        expected = [[0.,0.,0.,0.,1.], [0.5, 0.5, 0, 0.,0.], [0.,0.,0.5,0.5,0]]
        classifier.train()
        self.assertEqual(expected, classifier.posteriors)

    def test_predict(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        test_data = [['caresses'], ['feet cats foot'], ['talked wolves']]
        expected = ['caress', 'name', 'surprise']
        classifier.train()
        actual = classifier.predict(test_data)
        self.assertEqual(expected, actual)

    def test_predict_with_unexisting_words(self):
        data = np.array([['feet', 'body'], ['cats', 'animal'], ['wolves', 'animal'], ['mouth', 'body'], ['obama macron trudeau', 'politics']])
        classifier = NaiveBayesClassifier(data)
        test_data = [['foot foot mouth'], ['cats foot wolf wolf'], ['obama trudeau merkel']]
        expected = ['body', 'animal', 'animal']
        classifier.train()
        actual = classifier.predict(test_data)
        self.assertEqual(expected, actual)




