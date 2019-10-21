import random
from src.main.data_import import *
from datetime import datetime
random.seed(datetime.now())


def save_predictions(predictions):
    indexes = np.arange(0, len(predictions)).astype(str)
    indexes = np.insert(indexes, 0, ['Id'], 0)
    predictions = np.insert(predictions, 0, ['Category'], 0)
    preds = np.column_stack((indexes, predictions))
    np.savetxt("random_classifier_predictions.csv", preds, fmt="%s", delimiter=',')


def get_error(preds, labels):
    errors = np.array([preds[j] != labels[j] for j in range(len(preds))])
    error_count = errors.astype(int).sum()
    return error_count / len(preds)


class RandomClassifier:

    def __init__(self, train_data):
        self.train_data = train_data
        self.labels = np.unique(train_data[:, -1])

    def predict(self, test_data):
        return np.array([random.choice(self.labels) for j in range(len(test_data))])

    def predict_weighted(self, test_data):
        return np.array([random.choice(self.train_data[:,-1]) for j in range(len(test_data))])

