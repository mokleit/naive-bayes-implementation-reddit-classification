import unittest
from src.main.naive_bayes_classifier.naive_bayes_classifier import *
from src.main.data_import import *


class TestNaivesBayesClassifier(unittest.TestCase):

    #REMOVE ANNOTATION
    #AND EXECUTE ME
    def test_save_predictions(self):
        data_import = DataImport()
        data = get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        classifier = NaiveBayesClassifier(data, 0.59)
        test_data = data_import.get_test_data_as_list()
        classifier.train()
        predictions = classifier.predict(test_data)
        save_predictions(np.array(predictions), "smooth_naive_bayes_classifier_predictions.csv")

    def test_convert_to_sentence(self):
        data = ['Hello, you!', 'My name is Name Name.', 'What\'s yours?']
        expected = 'Hello, you! My name is Name Name. What\'s yours?'
        actual = convert_to_sentence(data)
        self.assertEqual(expected, actual)

    def test_remove_punctuation(self):
        sentence = 'Hello, you! My name is Name Name. What\'s yours?'
        expected = 'Hello you My name is Name Name Whats yours'
        actual = remove_punctuation(sentence)
        self.assertEqual(expected, actual)

    def test_define_vocabulary(self):
        data = np.array([['Feet! Cats, foot was,', 'name'], ['Cats my, wolf Was', 'name'], ['Wolves! talk', 'surprise!'], ['talked. caress', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        expected = ['caresses',
                    'feet',
                    'cats',
                    'foot',
                    'wolf',
                    'talked',
                    'caress',
                    'wolves',
                    'talk']
        classifier.train()
        self.assertEqual(expected, classifier.vocabulary)

    def test_split_data_by_label(self):
        data = np.array([['foot', 'body'], ['cat', 'animal'], ['obama', 'politics'], ['neck', 'body'], ['trudeau', 'politics']])
        classifier = NaiveBayesClassifier(data)
        expected = [
         [['cat']],
         [['foot'], ['neck']],
         [['obama'],['trudeau']]
        ]
        actual = classifier.split_data_by_label(data)
        actual = [arr.tolist() for arr in actual]
        self.assertEqual(expected, actual)

    def test_compute_priors(self):
        data = np.array([['feet', 'name'], ['cats', 'name'], ['wolves', 'surprise'], ['talked', 'surprise'], ['caresses', 'caress']])
        classifier = NaiveBayesClassifier(data)
        train_data_by_label = classifier.split_data_by_label(data)
        classifier.compute_priors(train_data_by_label)
        expected = ([1 / 5, 2 / 5, 2 / 5])
        self.assertEqual(expected, classifier.priors)

    def test_compute_word_frequencies_by_label(self):
        data = np.array([['feet cat', 'name'], ['cats wolf', 'name'], ['wolves', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        train_data_by_label = classifier.split_data_by_label(classifier.train_data)
        clean_train_data = classifier.prepare_train_data_for_posteriors(train_data_by_label)
        expected = [{'cat': 1, 'cats': 1, 'feet': 1, 'wolf': 1}, {'wolves': 1}]
        classifier.compute_words_frequency_by_label(clean_train_data)
        self.assertEqual(expected, classifier.words_frequency_by_label)

    def test_get_index_of_most_likely_label(self):
        data = np.array([['feet cat', 'name'], ['cats wolf', 'name'], ['wolves', 'surprise']])
        classifier = NaiveBayesClassifier(data)
        words = ['wolf', 'wolf']
        expected = 0
        classifier.train()
        actual = classifier.get_index_of_most_likely_label(words)
        self.assertEqual(expected, actual)

    def test_predict(self):
        data = np.array([['Foot neck ear', 'body'], ['ankle', 'body'], ['head leg', 'body'], ['eye', 'body'],
                         ['football', 'sports'], ['Tennis wrestling', 'sports'], ['golf', 'sports'], ['basketball', 'sports'],
                         ['trudeau', 'politics'], ['Macron', 'politics'], ['merkel', 'politics'], ['obama', 'politics']])
        classifier = NaiveBayesClassifier(data)
        test_data = ['Foot leg', 'leg Football golf golf', 'Macron']
        expected = ['body', 'sports', 'politics']
        classifier.train()
        actual = classifier.predict(test_data)
        self.assertEqual(expected, actual)

    def test_predict_with_unseen_words_in_test_data(self):
        data = np.array([['foot neck ear ankle ', 'body'], ['ankle neck', 'body'], ['head leg ear ankle', 'body'], ['eye foot leg head', 'body'],
                         ['football tennis', 'sports'], ['tennis wrestling', 'sports'], ['golf basketball', 'sports'], ['basketball wrestling', 'sports'],
                         ['trudeau macron', 'politics'], ['macron merkel', 'politics'], ['merkel obama trudeau', 'politics'], ['obama macron', 'politics']])
        classifier = NaiveBayesClassifier(data)
        test_data = ['Foot Foot neck hey ho hi', 'tennis Foot golf Basketball hgf', 'Obama Foot jhg Trudeau Merkel']
        expected = ['body', 'sports', 'politics']
        classifier.train()
        actual = classifier.predict(test_data)
        self.assertEqual(expected, actual)

