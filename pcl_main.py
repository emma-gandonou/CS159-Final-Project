##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import numpy as np
import argparse
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab, do_xml_parse
import xml.etree.ElementTree as ET

class BinaryLabels(PCLLabels):
    """extracts condescension attribute from an example"""

    def _extract_label(self, example):
        """gets value of the condescension attribute of the example"""
        return example.attrib["condescension"]


class CategoryLabels(PCLLabels):
    """extracts the category of the label from an example"""

    def _extract_label(self, example):
        """extracts the category of the example"""
        return example.attrib["category"]

class MyFeatures(PCLFeatures):
    """extracts features from example"""
    def __init__(self, vocab_file):
        super().__init__(vocab_file) # here vocab_file is a PCLVocab object
        self.vocab_words = set(vocab_file._words)

    def _get_feature_name(self, i):
        feature_names = self.vectorizer.feature_names_
        name = feature_names[i]
        return name

    def _extract_features(self, example):
        return [word for word in self.extract_text(example) if word in self.vocab_words]

    def _get_num_features(self):
        feature_names = self.vectorizer.feature_names_ 
        return len(feature_names)

def get_example_text_by_id(example_list, target_id):
    for element in example_list:
        if element.get('id') == target_id:
            return element.text
    return None


def do_experiment(args): 
    args.data_file.seek(0)
    example_list = list(do_xml_parse(args.data_file, 'example'))
    
    target_id = "@@1824078"  # Replace with your actual ID
    text = get_example_text_by_id(example_list, target_id)
    
    if text:
        print(f"Text for ID {target_id}: {text}")
    else:
        print(f"No example found with ID {target_id}")
    
    args.data_file.seek(0)
    # args.data_file.seek(0)
    
    # # Test the parser directly
    # parsed_examples = do_xml_parse(args.data_file, 'example')

    # counter = 1
    # while counter < 126:
    #     first_example = next(parsed_examples, None)
    #     counter += 1
    
    # if first_example is not None:
    #     print("First example found:")
    #     print(f"Tag: {first_example.tag}")
    #     print(f"Attributes: {first_example.attrib}")
    #     print(f"Text: {first_example.text}")
    # else:
    #     print("No examples found by parser!")
    
    # args.data_file.seek(0)
    
    # Now try the full list
    # args.data_file.seek(0)
    # example_list = list(do_xml_parse(args.data_file, 'example'))
    # print(f"Total examples parsed: {len(example_list)}")

    pcl_vocab = PCLVocab(args.vocabulary, args.vocab_size, args.stop_words)
    my_features = MyFeatures(pcl_vocab)
    binary_labels = BinaryLabels()
    category_labels = CategoryLabels()
    naive_bayes = MultinomialNB()
    # dummy = DummyClassifier(strategy = "uniform")

    data_file = args.data_file
    
    feature_matrix, example_ids = my_features.process(data_file)
    data_file.seek(0)
    label_matrix = binary_labels.process(data_file)
    data_file.seek(0)
    categories = category_labels.process(data_file)
    data_file.seek(0)

    if args.test_category is not None:
        category_list = np.array(category_labels._label_list)[categories]
        label_matrix = np.array(label_matrix)
        example_ids = np.array(example_ids)

        test_idx = np.where(args.test_category == category_list)[0]
        train_idx = np.where(category_list != args.test_category)[0]

        test_fdata = feature_matrix[test_idx]
        test_ldata = label_matrix[test_idx]
        train_fdata = feature_matrix[train_idx]
        train_ldata = label_matrix[train_idx]

        naive_bayes.fit(train_fdata, train_ldata)
        prediction = naive_bayes.predict(test_fdata)
        probabilities = naive_bayes.predict_proba(test_fdata)

        # dummy.fit(train_fdata, train_ldata)
        # prediction = dummy.predict(test_fdata)
        # probabilities = dummy.predict_proba(test_fdata)

        confidence = probabilities[np.arange(len(prediction)), prediction]    
        test_example_ids = example_ids[test_idx]

    else:
        prediction = cross_val_predict(naive_bayes, feature_matrix, label_matrix, cv=args.xvalidate, method="predict")
        probabilities = cross_val_predict(naive_bayes, feature_matrix, label_matrix, cv=args.xvalidate, method="predict_proba")

        # prediction = cross_val_predict(dummy, feature_matrix, label_matrix, cv=args.xvalidate, method="predict")
        # probabilities = cross_val_predict(dummy, feature_matrix, label_matrix, cv=args.xvalidate, method="predict_proba")

        confidence = probabilities[np.arange(len(prediction)), prediction]
        test_example_ids = example_ids

    for ex_id, pred, conf in zip(test_example_ids, prediction, confidence):
        pred_label = "true" if pred == 1 else "false"
        args.output_file.write(f"{ex_id} {pred_label} {conf:.6f}\n")


if __name__ == '__main__':                                                               
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=argparse.FileType('rb'), help="Data file containing labeled training instances")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="File containing vocabulary words")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=0 means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.output_file, args.vocabulary, args.data_file): fp.close()
