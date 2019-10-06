import numpy as np
import random
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
