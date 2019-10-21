import unittest
from src.main.naive_bayes_classifier_with_validation.naive_bayes_classifier_with_validation import *
from src.main.data_import import *
import csv


class TestNaivesBayesClassifierWithValidation(unittest.TestCase):

    def test_train(self):
        data_import = DataImport()
        data = data_import.get_clean_data_set_as_array(data_import.get_train_data_as_tuple())
        classifier = NaiveBayesClassifierWithValidation(data, 0.59)
        error_rate, mis_indices, mis_labels, words_freq, test = classifier.train()
        print('Error rate', error_rate)
        print('Accuracy', 1. - error_rate)
        print("CREATING FILES")
        for i in range(20):
            w = csv.writer(open("dictionary_%s.csv" % str(i+1), "w"))
            for key, val in words_freq[i].items():
                w.writerow([key, val])

        mis_test_data = test[mis_indices]
        temp_test_data = np.hstack(mis_test_data[:, :-1])
        for i in range(len(mis_test_data)):
            cleaned = classifier.clean_test_data(temp_test_data[i])
            mis_test_data[i, :-1] = NaiveBayesClassifier.convert_to_sentence(cleaned)
        print(mis_test_data)
        np.savetxt('misclassified_test_data.csv', mis_test_data, fmt="%s", delimiter=',')
        test_mis_label_comp = np.insert(mis_test_data, mis_test_data.shape[1], mis_labels, axis=1)
        np.savetxt('misclassified_examples.csv', test_mis_label_comp, fmt="%s", delimiter=',')







