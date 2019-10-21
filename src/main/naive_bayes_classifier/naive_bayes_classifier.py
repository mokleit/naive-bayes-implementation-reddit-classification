import numpy as np
import nltk as nlt
import string
from collections import Counter
from nltk.corpus import stopwords


class NaiveBayesClassifier:

    def __init__(self, train_data, alpha=0.59):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])
        self.number_of_labels = len(self.labels)
        self.vocabulary = []
        self.vocabulary_length = 0
        self.priors = initialize_list(self.number_of_labels, 0)
        self.words_frequency_by_label = initialize_list(self.number_of_labels, 0)
        self.dictionaries_length = initialize_list(self.number_of_labels, 0)
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = nlt.tokenize.TreebankWordTokenizer()
        self.alpha = alpha

    def train(self):
        train_data_by_label = self.split_data_by_label(self.train_data)
        self.compute_priors(train_data_by_label)
        clean_train_data = self.prepare_train_data_for_posteriors(train_data_by_label)
        self.define_vocabulary(clean_train_data)
        self.compute_words_frequency_by_label(clean_train_data)

    def predict(self, test_data):
        preds = initialize_list(len(test_data), '')
        for i in range(len(test_data)):
            words = self.clean_test_data(test_data[i])
            index = self.get_index_of_most_likely_label(words)
            preds[i] = self.labels[index]
        return preds

    def get_index_of_most_likely_label(self, words):
        likelihoods = initialize_list(self.number_of_labels, 0)
        for i in range(self.number_of_labels):
            posterior = np.log(self.priors[i])
            for word in words:
                posterior += self.compute_word_posterior(i, word)
            likelihoods[i] = posterior
        return np.argmax(likelihoods)

    def compute_word_posterior(self, index, existing_word):
        dictionnary = self.words_frequency_by_label[index]
        try:
            return np.log((dictionnary[existing_word] + self.alpha) / (self.dictionaries_length[index] + (self.alpha * self.vocabulary_length)))
        except:
            return np.log(self.alpha / (self.dictionaries_length[index] + (self.alpha * self.vocabulary_length)))

    def compute_words_frequency_by_label(self, clean_train_data):
        for i in range(len(clean_train_data)):
            self.words_frequency_by_label[i] = dict(Counter(clean_train_data[i]))
            self.dictionaries_length[i] = sum(self.words_frequency_by_label[i].values())

    def compute_priors(self, train_data_by_label):
        total_number_of_docs = 0
        for i in range(self.number_of_labels):
            total_number_of_docs += len(train_data_by_label[i])
        for i in range(self.number_of_labels):
            self.priors[i] += (len(train_data_by_label[i]) / total_number_of_docs)

    def prepare_train_data_for_posteriors(self, train_data_by_label):
        for i in range(self.number_of_labels):
            train_data_by_label[i] = self.clean_and_tokenize(train_data_by_label[i])
        return train_data_by_label

    def split_data_by_label(self, data):
        split_data = [data[data[:,-1] == self.labels[i], :-1] for i in range(self.number_of_labels)]
        return split_data

    def clean_and_tokenize(self, label_train_data):
        sentence = convert_to_sentence(label_train_data)
        no_punctuation = remove_punctuation(sentence)
        lowered_words = no_punctuation.lower()
        tokens = self.tokenizer.tokenize(lowered_words)
        words = [word for word in tokens if word not in self.stopwords]
        return words

    def clean_test_data(self, test_example):
        no_punctuation = remove_punctuation(test_example)
        lowered_words = no_punctuation.lower()
        tokens = self.tokenizer.tokenize(lowered_words)
        words = [word for word in tokens if word in self.vocabulary]
        return words

    def define_vocabulary(self, unflattened_vocabulary):
        self.vocabulary = process_vocabulary(unflattened_vocabulary)
        self.vocabulary_length = len(self.vocabulary)


def convert_to_sentence(label_train_data):
    words_for_label = np.hstack(label_train_data)
    return ' '.join(words_for_label)


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def process_vocabulary(unfiltered_vocabulary_as_array):
    vocabulary = np.hstack(unfiltered_vocabulary_as_array)
    words = [word.lower() for word in vocabulary]
    return list(dict.fromkeys(words))


def initialize_list(size, value):
    return [value] * size


def save_predictions(predictions, filename="naive_bayes_classifier_predictions.csv"):
    indexes = np.arange(0, len(predictions)).astype(str)
    indexes = np.insert(indexes, 0, 'Id', 0)
    predictions = np.insert(predictions, 0, 'Category', 0)
    preds = np.column_stack((indexes, predictions))
    np.savetxt(filename, preds, fmt="%s", delimiter=',')
