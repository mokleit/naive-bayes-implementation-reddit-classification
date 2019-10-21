import numpy as np
from src.main.naive_bayes_classifier.naive_bayes_classifier import *


class NaiveBayesClassifierWithValidation(NaiveBayesClassifier):

    def train(self):
        print("TRAINING...")
        train, test = self.split_train_validation_data(self.train_data)
        train_data_by_label = self.split_data_by_label(train)
        self.compute_priors(train_data_by_label)
        clean_train_data = self.prepare_train_data_for_posteriors(train_data_by_label)
        self.define_vocabulary(clean_train_data)
        self.compute_words_frequency_by_label(clean_train_data)
        print("VALIDATING...")
        error_rate, mis_ind, mis_labels = self.validate_training(test)
        return error_rate, mis_ind, mis_labels, self.words_frequency_by_label, test

    def validate_training(self, validation_data):
        preds = self.predict(np.hstack(validation_data[:, :-1]))
        labels = validation_data[:,-1]
        errors = (np.array(preds) != np.array(labels)).astype(int)
        misclassified_indices = [i for i in range(len(errors)) if errors[i] == 1]
        misclassified_labels = [preds[i] for i in range(len(errors)) if i in misclassified_indices]
        return errors.sum() / len(preds), misclassified_indices, misclassified_labels

    def split_train_validation_data(self, train_data):
        train_len = len(train_data)
        train_quarter = int(0.25*train_len)
        indices1 = np.arange(0, train_quarter)
        indices2 = np.arange(train_quarter, 2 * train_quarter)
        indices3 = np.arange(2 * train_quarter, 3 * train_quarter)
        indices4 = np.arange(3 * train_quarter, 4 * train_quarter)

        train_proportion = int(0.98*train_quarter)

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
        return train, test

