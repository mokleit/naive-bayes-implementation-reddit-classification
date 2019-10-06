from unittest import TestCase
from src.main.data_import import *

class TestDataImport(TestCase):

    def test_get_train_data_as_tuple(self):
        train_data = DataImport().get_train_data_as_tuple()
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(train_data[0]), 70000)
        self.assertIs(type(train_data), tuple)

    def test_get_test_data_as_list(self):
        test_data = DataImport().get_test_data_as_list()
        self.assertEqual(len(test_data), 30000)
        self.assertIs(type(test_data), list)

    def test_convert_to_array(self):
        tuple = [(1,2,3,4), (5,6,7,8)]
        actual = DataImport().convert_to_array(tuple)
        self.assertEqual(actual.shape, (2, 4))
        self.assertIs(type(actual), np.ndarray)

    def test_get_labels(self):
        list = [2,1,3,3,4,5,1,2,3,5,4,3,2,1]
        actual = DataImport().get_labels(list)
        expected = [1,2,3,4,5]
        np.testing.assert_array_equal(actual, expected)

    def test_get_clean_data_set(self):
        data_import = DataImport()
        train_data = data_import.get_train_data_as_tuple()
        actual = data_import.get_clean_data_set_as_array(train_data)
        self.assertEqual(actual.shape, (70000, 2))
        self.assertIs(type(actual), np.ndarray)




