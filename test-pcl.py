import unittest
import numpy as np
from pcl_main import *

class TestPCL(unittest.TestCase):

    TEST_DATA_FILE = "/courses/cs159/data/patronize/patronize_full.xml"
    TEST_VOCAB = "/courses/cs159/data/patronize/vocab.txt"

    def testBinaryLabels(self):
        fp = open(self.TEST_DATA_FILE, 'rb')
        binary_labeler = BinaryLabels()
        small_labels = binary_labeler.process(fp, max_instances=10)
        self.assertEqual(small_labels, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        fp.seek(0)
        large_labels = binary_labeler.process(fp, max_instances=500)
        self.assertEqual(np.sum(large_labels), 34)
        fp.close()

    def testCategoryLabels(self):
        fp = open(self.TEST_DATA_FILE, 'rb')
        category_labeler = CategoryLabels()
        small_cats = category_labeler.process(fp, max_instances=10)
        self.assertEqual(small_cats, [2, 1, 2, 3, 4, 5, 4, 0, 3, 0])
        fp.close()

    def testMyFeatures(self):
        vocab_file = open(self.TEST_VOCAB, 'r')
        vocabulary = PCLVocab(vocab_file, 10, 0)
        features = MyFeatures(vocabulary)
        data_file = open(self.TEST_DATA_FILE, 'rb')
        X, _ = features.process(data_file, max_instances=1)
        final_vocab = [features._get_feature_name(i) for i in range(X.shape[1])]
        self.assertTrue('.' in final_vocab)
        self.assertTrue('and' in final_vocab)
        self.assertTrue('in' in final_vocab)
        self.assertTrue('of' in final_vocab)
        self.assertTrue('the' in final_vocab)
        self.assertFalse('that' in final_vocab)
        self.assertFalse('have' in final_vocab)

        vocab_file.seek(0)
        data_file.seek(0)

        vocabulary = PCLVocab(vocab_file, 10, 5)
        features = MyFeatures(vocabulary)
        X, _ = features.process(data_file, max_instances=1)
        final_vocab = [features._get_feature_name(i) for i in range(X.shape[1])]
        self.assertTrue('and' in final_vocab)
        self.assertTrue('are' in final_vocab)
        self.assertTrue('in' in final_vocab)
        self.assertTrue('on' in final_vocab)
        self.assertTrue('that' in final_vocab)
        self.assertFalse('.' in final_vocab)
        self.assertFalse('of' in final_vocab)
        self.assertFalse('the' in final_vocab)
        self.assertFalse('have' in final_vocab)

        vocab_file.close()
        data_file.close()

if __name__ == '__main__':
    unittest.main()