import numpy as np
import nltk as nlt
nlt.download('wordnet')


class NaiveBayesClassifier:

    def __init__(self, train_data):
        self.train_data = train_data[:, :-1]
        self.labels = train_data[:, -1]
        self.vocabulary = self.extract_vocabulary(self.train_data)


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

