import random
from src.main.data_import import *
from datetime import datetime
random.seed(datetime.now())

class RandomClassifier:

    def __init__(self, train_data):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])

    def predict(self, test_data):
        return np.array([random.choice(self.labels) for j in range(len(test_data))])

    def get_error(self, test_data):
        preds = self.predict(test_data[:, :-1])
        labels = test_data[:, -1]
        errors = np.array([preds[j] != labels[j] for j in range(len(preds))])
        error_count = errors.astype(int).sum()
        return error_count / len(preds)

    def save_predictions(self, test_data):
        train_data = self.train_data
        test_data = test_data

        predictions = RandomClassifier(train_data).predict(test_data)
        indexes = np.arange(0, len(predictions)).astype(str)
        indexes = np.insert(indexes, 0, ['Id'], 0)
        predictions = np.insert(predictions, 0, ['Category'], 0)

        preds = np.column_stack((indexes, predictions))

        for (i, row) in enumerate(preds):
            row[0] += ','

        np.savetxt("random_classifier_predictions.csv", preds, fmt="%s")
