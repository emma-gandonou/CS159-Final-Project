"""Calculates the measures for the PCL 2022 patronizing language detection task"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from xml.etree.ElementTree import iterparse

def main(args):
    groundTruth = {}

    for _, elem in iterparse(args.inputDataset):
        if elem.tag != 'example': continue
        groundTruth[elem.get('id')] = elem.get('condescension')
        elem.clear()
            
    c = Counter()

    for line in args.inputRun: 
        values = line.rstrip('\n').split()
        exampleId, prediction = values[:2]
            
        c[(prediction, groundTruth[exampleId])] += 1

    if sum(c.values()) < len(groundTruth):
        warnings.warn("Missing {} predictions".format(len(groundTruth) - sum(c.values())), UserWarning)
        
    tp = c[('true', 'true')]
    tn = c[('false', 'false')]
    fp = c[('true', 'false')]
    fn = c[('false', 'true')]
    
    accuracy  = (tp + tn) / sum(c.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {"truePositives": tp, "trueNegatives": tn, "falsePositives": fp, "falseNegatives": fn,
               "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    json.dump(results, args.outputFile, indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--inputDataset", type=argparse.FileType('rb'), required=True)
    parser.add_argument("-r", "--inputRun", type=argparse.FileType('r'), required=True)
    parser.add_argument("-o", "--outputFile", type=argparse.FileType('w'), default=sys.stdout)

    args=parser.parse_args()

    main(args)

    args.outputFile.close()
    