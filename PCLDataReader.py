##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from collections import Counter
from html import unescape
from lxml import etree
from sklearn.feature_extraction import DictVectorizer
import sys
import pandas


#####################################################################
# HELPER FUNCTIONS
#####################################################################

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """ 
    Parses cleaned up spacy-processed XML files
    param fp: filepath (file object)
    param tag: tag of example (string)
    param max_elements: the maximum number of elements returned (int)
    param progress_message: (string)
    return: each example in fp and progress message
    """
    fp = pandas.read_csv('archive.zip/books_and_genres.csv')

    # fp.seek(0)
    fp.reset_index(drop=True, inplace=True)


    # grabs [max_elements] number of elements from fp with desired tag, and returns it with an index 
    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))

    for i, (event, elem) in elements:
        yield elem
        elem.clear()

        # displaying progress of working through elements every 1000 elements
        if progress_message and (i % 1000 == 0): 
            print(progress_message.format(i), file=sys.stderr, end='\r')
    
    # when all elements have been returned, prints the system error stream 
    if progress_message: print(file=sys.stderr)

def short_xml_parse(fp, tag, max_elements=None): 
    """ 
    Parses cleaned up spacy-processed XML files (but not very well)
    param fp: file object (object)
    param tag: desired tag of object (string)
    param max_elements: max elements to return (int)
    return: list of examples from fp
    """

    # gets all the elements with the associated tag
    elements = etree.parse(fp).findall(tag)

    # ensure that if max_elements is specified, then that number of max elements is returned 
    N = max_elements if max_elements is not None else len(elements)
    return elements[:N]

#####################################################################
# PCLVocab
#####################################################################

class PCLVocab(): 
    def __init__(self, vocab_file, vocab_size, num_stop_words): 
        """
        takes a file and extracts words from the desired segments, creating a dictionary of word and index 
        param vocab_file: a list containing all the vocab words (list)
        param vocab_size: how many vocabs to consider (int)
        param num_stop_words: starting point in vocab files (int)
        """
        start_index = 0 if num_stop_words is None else num_stop_words
        end_index = start_index + vocab_size if vocab_size is not None else None

        # grab words from desired segment and extracts words from it 
        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)]

        # flipping enumerate so that the dict as the word has key and index as value 
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)])

    def __len__(self): 
        """
        returns number of words in the object
        """
        return len(self._dict)

    def index_to_label(self, i): 
        """
        returns the label of a certain index in the object
        """
        return self._words[i]

    def __getitem__(self, key):
        """
        returns the value of a certain key in the object
        returns none is key isn't in the vocab
        """
        if key in self._dict: return self._dict[key]
        else: return None

#####################################################################
# PCLLabels
#####################################################################

class PCLLabels(ABC):
    def __init__(self): 
        """
        initiates a PCLLabels object with variables labels and _label_list
        """
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        """
        extracts all the labels from the inputted file 
        param label_file: the file to be processed (xml file object)
        param max_instances: optional number of elements to consider from the file 
        returns: all the index of labels from the text in the file (list of ints)
        """

        # gets the text of each label in _extract_label
        y_labeled = list(map(self._extract_label, do_xml_parse(label_file, 'example', max_elements=max_instances)))
        
        # if there's no specified labels, then set labels based on unique labels from _label_list
        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])
        
        # gets all the indices of labels from the file 
        y = [self.labels[x] for x in y_labeled]
        return y

    @abstractmethod        
    def _extract_label(self, example):
        """ Return the label for this instance """
        return "Unknown"

#####################################################################
# PCLFeatures
#####################################################################

class PCLFeatures(ABC): 
    def __init__(self, vocab):
        """
        initializes object with an inputted vocab and a sparse feature matrix 
        param vocab: list of tokens (list)
        """
        self.initial_vocab = vocab
        self.vectorizer = DictVectorizer(sparse=True) # representation of spare vector where most values are 0

    def extract_text(self, example):
        """ 
        param example: the desired xml file 
        returns: list of lower cased words from the file 
        """
        # cleaning the text to be all lower and removing html tags, returning list of words
        return unescape("".join([x for x in example.itertext()]).lower()).split()

    def process(self, data_file, max_instances=None):
        """
        param data_file: desired path (file object)
        param max_instances: maximum examples to consider from file (int)
        return: a feature matrix based on the examples and IDs for each row 
        """
        # if there isn't anything set for max_instances, set it to how many examples are in the data file
        if max_instances == None:
            N = len([1 for example in do_xml_parse(data_file, 'example')])
        else:
            N = max_instances
        
        # make a list full of ids in each example from data file
        # make a list full of Counters of features from each example in data file
        ids = []
        feature_counters = []

        # for each example, extracts the features and counts how many there are
        for example in do_xml_parse(data_file, 'example', max_elements=N, progress_message="Example {}"):
            ids.append(example.get("id"))
            features = self._extract_features(example)
            feature_counters.append(Counter(features))
        
        # using the feature counts for each example, creates a sparse feature vector 
        X = self.vectorizer.fit_transform(feature_counters)
        return X, ids

    @abstractmethod
    def _get_feature_name(self, i):
        """ Returns a human-readable name for the ith feature in the DictVectorizer's internal vocabulary """
        return "Unknown"

    @abstractmethod            
    def _extract_features(self, example):
        """ Returns a list of the features in the example """
        return []

    @abstractmethod        
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################