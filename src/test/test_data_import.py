from unittest import TestCase
from src.main.data_import import *

class TestDataImport(TestCase):

    def test_get_train_data(self):
        train_data = DataImport().get_train_data()
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(train_data[0]), 70000)

    def test_get_test_data(self):
        test_data = DataImport().get_test_data()
        self.assertEqual(len(test_data), 30000)

    def test_convert_to_array(self):
        tuple = [(1,2,3,4), (5,6,7,8)]
        actual = DataImport().convert_to_array(tuple)
        self.assertEqual(actual.shape, (2, 4))



