#!/usr/bin/env python
import argparse
import codecs
from sklearn.externals import joblib
from util import build_features, test_svm, compute_metrics, test_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NER on SVM')
    parser.add_argument('-t', '--test', default='../test-raw/', help='test file')
    parser.add_argument('-r', '--test_ref', default='../test/', help='reference labels test file')
    parser.add_argument('-m', '--model', default='../models/trained_full_sfx_pfx_0506_2040.pkl', help='Model Trained with SVM')
    opts = parser.parse_args()
    labels_test = []
    predict_test = []
    features_test = []
    clf = joblib.load(opts.model)
    print "Building Features with Test Set..."
    build_features(opts.test, opts.test_ref, features_test, labels_test,1)
    print "Predict with Trained Model..."
    test_svm(features_test, predict_test, clf)
    print "Compute Metrics..."
    compute_metrics(predict_test, labels_test)
    with codecs.open('NE.hi',encoding="utf8",mode="wb") as o_file:
        for ind in range(len(predict_test[0])):
            if predict_test[0][ind]==1:     #or ind in test_words_in_train:
                o_file.write(test_words[ind])
                o_file.write('\n')
    o_file.close()