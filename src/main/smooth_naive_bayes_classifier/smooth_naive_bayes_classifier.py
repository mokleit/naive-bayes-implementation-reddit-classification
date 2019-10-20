from src.main.naive_bayes_classifier.naive_bayes_classifier import *

class SmoothNaiveBayesClassifier(NaiveBayesClassifier):

    def compute_posteriors(self, doc_label, label_index):
        words = self.tokenize_sentence(doc_label)
        words_count = len(words)
        no_dup_words = list(dict.fromkeys(words))
        for i in range(len(no_dup_words)):
            count = words.count(no_dup_words[i])
            word_index = self.vocabulary.index(no_dup_words[i])
            self.posteriors[label_index][word_index] = (count + 1) / (words_count + len(self.vocabulary))
        self.smooth_laplace(words_count)

    def smooth_laplace(self, words_count):
        for i in range(self.number_of_labels):
            for j in range(len(self.vocabulary)):
                if self.posteriors[i][j] == 0.0:
                    self.posteriors[i][j] = 1. / (words_count + len(self.vocabulary))

