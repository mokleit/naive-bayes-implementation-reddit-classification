import numpy as np
import nltk as nlt


class NaiveBayesClassifier:

    def __init__(self, train_data):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])
        self.number_of_labels = len(self.labels)

    def get_priors(self, docs_number_by_label, total_docs_number):
        priors = self.initialize_list(self.number_of_labels, 0.)
        for i in range(self.number_of_labels):
            priors[i] += docs_number_by_label[i] / total_docs_number
        return priors

    def get_docs_by_label(self):
        docs = self.initialize_list(self.number_of_labels, '')
        docs_number = self.initialize_list(self.number_of_labels, 0)
        for i in range(len(self.train_data)):
            j = 0
            while True:
                if self.train_data[i,-1] == self.labels[j]:
                    sentence = self.convert_to_sentence(self.train_data[i,:-1])
                    if len(docs[j]) > 0:
                        docs[j] += ' '
                    docs[j] += sentence
                    docs_number[j] += 1
                    break
                else:
                    j += 1
        return docs, docs_number

    def initialize_list(self, size, value):
        my_list = []
        for i in range(size):
            my_list.append(value)
        return my_list

    def extract_vocabulary(self, data):
        sentence = self.convert_to_sentence(data)
        tokenizer = nlt.tokenize.TreebankWordTokenizer()
        words = tokenizer.tokenize(sentence)
        clean_words = self.clean_unfiltered_punctuation(words)
        vocabulary = self.stem_words(clean_words)
        return vocabulary

    def convert_to_sentence(self, message_array):
        flatten_data = np.ndarray.flatten(message_array).astype(str)
        text = ' '.join(flatten_data)
        return text

    def clean_unfiltered_punctuation(self, words):
        for (i, word) in enumerate(words):
            if len(word) > 1 and word.endswith(('.', ',', '!', '?', ':', ';')):
                words[i] = word[:-1]
                words.insert(i+1, word[-1])
        return words

    def stem_words(self, words):
        stemmer = nlt.stem.WordNetLemmatizer()
        stemmed = [stemmer.lemmatize(word) for word in words]
        return list(stemmed)

