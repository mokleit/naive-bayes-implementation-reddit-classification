import numpy as np
import nltk as nlt

class NaiveBayesClassifier:

    def __init__(self, train_data):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])
        self.number_of_labels = len(self.labels)
        self.vocabulary = self.extract_vocabulary(self.train_data[:, :-1])
        self.priors = self.initialize_list(self.number_of_labels, 0.)
        self.posteriors = self.initialize_2d_list(self.number_of_labels, len(self.vocabulary), 0.)

    def train(self):
        #Get docs and docs number by label
        docs_by_label, docs_number_by_label = self.get_docs_by_label()

        #Calculate p(cj) terms
        total_docs_number = np.array(docs_number_by_label).sum()
        self.compute_priors(docs_number_by_label, total_docs_number)

        #Calculate p(wk|cj) terms
        for i in range(len(docs_by_label)):
            self.compute_posteriors(docs_by_label[i], i)

    def predict(self, test_data):
        preds = self.initialize_list(len(test_data), '')
        for i in range(len(test_data)):
            tokens = self.sentence_to_cleaned_words_list(''.join(test_data[i]))
            pred_label_index = self.get_most_likely_label(tokens)
            preds[i] += self.labels[pred_label_index]
        return preds

    def get_most_likely_label(self, words):
        likelihood = self.initialize_list(self.number_of_labels, 0.)
        for i in range(self.number_of_labels):
            prior = self.priors[i]
            posteriors = 1
            for j in range(len(words)):
                unknown = False
                try:
                    index = self.vocabulary.index(words[j])
                except:
                    unknown = True
                if unknown == False:
                    posteriors *= self.posteriors[i][index]
            likelihood[i] += (prior * posteriors)
        return np.argmax(np.array(likelihood))

    def compute_posteriors(self, doc_label, label_index):
        words = self.sentence_to_cleaned_words_list(doc_label)
        words_count = len(words)
        no_dup_words = list(dict.fromkeys(words))
        for i in range(len(no_dup_words)):
            count = words.count(no_dup_words[i])
            word_index = self.vocabulary.index(no_dup_words[i])
            self.posteriors[label_index][word_index] = count / words_count

    def compute_priors(self, docs_number_by_label, total_docs_number):
        for i in range(self.number_of_labels):
            self.priors[i] += docs_number_by_label[i] / total_docs_number

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
        for j in range(len(docs)):
            cleaned_list = list(dict.fromkeys(self.sentence_to_cleaned_words_list(docs[j])))
            docs[j] = ' '.join(cleaned_list)
        return docs, docs_number

    def initialize_list(self, size, value):
        my_list = []
        for i in range(size):
            my_list.append(value)
        return my_list

    def initialize_2d_list(self, rows, columns, value):
        my_list = []
        for i in range(rows):
            inner_list = self.initialize_list(columns, value)
            my_list.append(inner_list)
        return my_list

    def extract_vocabulary(self, data):
        sentence = self.convert_to_sentence(data)
        vocabulary = self.sentence_to_cleaned_words_list(sentence)
        return list(dict.fromkeys(vocabulary))

    def sentence_to_cleaned_words_list(self, sentence):
        tokenizer = nlt.tokenize.TreebankWordTokenizer()
        words = tokenizer.tokenize(sentence)
        clean_words = self.clean_unfiltered_punctuation(words)
        tokens = self.stem_words(clean_words)
        return tokens

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


    def save_predictions(self, predictions, filename="naive_bayes_classifier_predictions.csv"):
        indexes = np.arange(0, len(predictions)).astype(str)
        indexes = np.insert(indexes, 0, ['Id'], 0)
        predictions = np.insert(predictions, 0, ['Category'], 0)

        preds = np.column_stack((indexes, predictions))

        np.savetxt(filename, preds, fmt="%s", delimiter=',')
