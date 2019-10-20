import numpy as np
import nltk as nlt
import time
import string
from collections import Counter
from nltk.corpus import stopwords

class NaiveBayesClassifier:

    def __init__(self, train_data):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])
        self.number_of_labels = len(self.labels)
        self.vocabulary = []
        self.vocabulary_length = 0
        self.priors = self.initialize_list(self.number_of_labels, 0)
        self.word_frequencies_by_label = self.initialize_list(self.number_of_labels, 0)#create list of dictionaries
        self.dictionaries_length = self.initialize_list(self.number_of_labels, 0)
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = nlt.tokenize.TreebankWordTokenizer()
        self.wordnet_lemmatizer = nlt.stem.WordNetLemmatizer()
        self.snowball_stemmer = nlt.stem.SnowballStemmer('english')
        self.smoothing = 0.58


    def train(self):
        t1 = time.process_time()
        train, test = self.split_train_validation_data(self.train_data)
        #print('VALIDATION SET', test)
        train_data_by_label = self.split_data_by_label(train)
        self.compute_priors(train_data_by_label)
        clean_train_data = self.prepare_train_data_for_posteriors(train_data_by_label)
        self.define_vocabulary(clean_train_data)
        #print('VOCAB', self.vocabulary)
        self.vocabulary_length = len(self.vocabulary)
        self.compute_word_frequencies_by_label(clean_train_data)
        t2 = time.process_time()
        print('TRAINING', t2-t1, 'seconds')
        t3 = time.process_time()
        error_rate = self.validate_training(test)
        t4 = time.process_time()
        print('VALIDATION', t4-t3, 'seconds')
        print("ERROR RATE:", error_rate)
        print("ACCURACY:", 1. - error_rate)

    ################### BEG TRAINING #####################################################################
    def split_data_by_label(self, data):
        split_data = [data[data[:,-1] == self.labels[i], :-1] for i in range(self.number_of_labels)]
        return split_data

    def compute_priors(self, train_data_by_label):
        t3 = time.process_time()
        total_number_of_docs = 0
        for i in range(self.number_of_labels):
            total_number_of_docs += len(train_data_by_label[i])
        for i in range(self.number_of_labels):
            self.priors[i] += (len(train_data_by_label[i]) / total_number_of_docs)
        t4 = time.process_time()
        print("COMPUTE PRIORS", t4-t3, 'seconds')

    def prepare_train_data_for_posteriors(self, train_data_by_label):
        t6 = time.process_time()
        for i in range(self.number_of_labels):
            train_data_by_label[i] = self.clean_and_tokenize(train_data_by_label[i])
        t7 = time.process_time()
        print('CLEAN TRAIN DATA', t7-t6, 'seconds')
        return train_data_by_label

    def clean_and_tokenize(self, label_train_data):
        sentence = self.convert_to_sentence(label_train_data)
        no_punctuation = self.remove_punctuation(sentence)
        lowered_words = no_punctuation.lower()
        tokens = self.tokenizer.tokenize(lowered_words)
        #stemmed_words = [self.snowball_stemmer.stem(word) for word in tokens if word not in self.stopwords]
        lemmatized_words = [self.wordnet_lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        #words = [word for word in tokens if word not in self.stopwords]
        return lemmatized_words

    def convert_to_sentence(self, label_train_data):
        words_for_label = np.hstack(label_train_data)
        return ' '.join(words_for_label)

    def remove_punctuation(self, sentence):
        return sentence.translate(str.maketrans('', '', string.punctuation))

    def define_vocabulary(self, unflattened_vocabulary):
        t8 = time.process_time()
        self.vocabulary = self.filter_vocabulary(unflattened_vocabulary)
        t9 = time.process_time()
        print('DEFINE VOCABULARY', t9-t8, 'seconds')

    def filter_vocabulary(self, unfiltered_vocabulary_as_array):
        vocabulary = np.hstack(unfiltered_vocabulary_as_array)
        words = [word.lower() for word in vocabulary]
        return list(dict.fromkeys(words))

    def compute_word_frequencies_by_label(self, clean_train_data):
        t10 = time.process_time()
        for i in range(len(clean_train_data)):
            self.word_frequencies_by_label[i] = dict(Counter(clean_train_data[i]))
            self.dictionaries_length[i] = sum(self.word_frequencies_by_label[i].values())
        t11 = time.process_time()
        #print('STORE WORD FREQUENCIES IN DICT', t11 - t10, 'seconds')
##################### END TRAINING #####################################################################

##################### BEG VALIDATION #####################################################################
    def split_train_validation_data(self, train_data):
        beg_split = time.process_time()
        train_len = len(train_data)
        train_quarter = int(0.25*train_len)
        indices1 = np.arange(0, train_quarter)
        indices2 = np.arange(train_quarter, 2 * train_quarter)
        indices3 = np.arange(2 * train_quarter, 3 * train_quarter)
        indices4 = np.arange(3 * train_quarter, 4 * train_quarter)
        #print('INDICES')
        #print(indices1)
        #print(indices2)
        #print(indices3)
        #print(indices4)

        #np.random.shuffle(indices1)
        #np.random.shuffle(indices2)
        #np.random.shuffle(indices3)
        #np.random.shuffle(indices4)

        train_proportion = int(0.98*train_quarter)
        #print('train proportion', train_proportion)

        train1 = train_data[indices1[:train_proportion]]
        test1 = train_data[indices1[train_proportion:]]
        train2 = train_data[indices2[:train_proportion]]
        test2 = train_data[indices2[train_proportion:]]
        train3 = train_data[indices3[:train_proportion]]
        test3 = train_data[indices3[train_proportion:]]
        train4 = train_data[indices4[:train_proportion]]
        test4 = train_data[indices4[train_proportion:]]

        train = np.concatenate([train1, train2, train3, train4])
        test = np.concatenate([test1, test2, test3, test4])
        end_split = time.process_time()
        #print('train', train)
        #print('test', test)
        print("SPLIT TRAIN/VAL:", end_split-beg_split, 'seconds')
        return train, test

    def validate_training(self, validation_data):
        preds = self.predict(np.hstack(validation_data[:, :-1]))
        labels = validation_data[:,-1]
        errors = (np.array(preds) != np.array(labels)).astype(int).sum()
        return errors / len(preds)
##################### END VALIDATION #######################################################

##################### BEG PREDICT #######################################################
    def predict(self, test_data):
        preds = self.initialize_list(len(test_data), '')
        print("PREDICTING...")
        #start = time.process_time()
        #test_data = [self.clean_test_data(example) for example in test_data]
        end = time.process_time()
        #print("TOOK:", end-start, 'seconds')
        t12 = time.process_time()
        for i in range(len(test_data)):
            #print("CLEANING TEST EXAMPLE", i)
            start = time.process_time()
            no_punctuation = self.remove_punctuation(test_data[i])
            #print('no punctuation word', no_punctuation)
            lowered_words = no_punctuation.lower()
            #print('lowered words', lowered_words)
            tokens = self.tokenizer.tokenize(lowered_words)
            #print('tokens', tokens)
            #stemmed_words = [self.snowball_stemmer.stem(word) for word in tokens]
            #print('stemmed words', stemmed_words)
            lemmatized_words = [self.wordnet_lemmatizer.lemmatize(word) for word in tokens]
            t15 = time.process_time()
            words = [word for word in lemmatized_words if word in self.vocabulary]
            t16 = time.process_time()
            #print('Filtering test data from vocab took', t16-t15, 'seconds')
            #print('TEST DATA', i, ':', lemmatized_words)
            end = time.process_time()
            #print("TOOK:", end-start, 'seconds')
            t3 = time.process_time()
            index = self.get_index_of_most_likely_label(words)
            t4 = time.process_time()
            #print('EXAMPLE:', i , 'LABEL:', self.labels[index], 'TIME:', t4-t3, 'seconds')
            preds[i] = self.labels[index]
        t13 = time.process_time()
        print("COMPUTE PREDS", t13-t12, 'seconds')
        return preds

    def get_index_of_most_likely_label(self, words):
        start_init = time.process_time()
        likelihoods = self.initialize_list(self.number_of_labels, 0)
        end_init = time.process_time()
        for i in range(self.number_of_labels):
            posterior = np.log(self.priors[i])
            for word in words:
                posterior += self.compute_word_posterior(i, word)
            likelihoods[i] = posterior
        return np.argmax(likelihoods)

    def compute_word_posterior(self, index, existing_word):
        dictionnary = self.word_frequencies_by_label[index]
        try:
            return np.log((dictionnary[existing_word] + self.smoothing) / (self.dictionaries_length[index] + (self.smoothing * self.vocabulary_length)))
        except:
            return np.log(self.smoothing / (self.dictionaries_length[index] + (self.smoothing * self.vocabulary_length)))
########## END PREDICT #######################################################

    def initialize_list(self, size, value):
        return [value] * size

    def save_predictions(self, predictions, filename="naive_bayes_classifier_predictions.csv"):
        indexes = np.arange(0, len(predictions)).astype(str)
        indexes = np.insert(indexes, 0, 'Id', 0)
        predictions = np.insert(predictions, 0, 'Category', 0)

        preds = np.column_stack((indexes, predictions))

        np.savetxt(filename, preds, fmt="%s", delimiter=',')
