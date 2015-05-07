#!/usr/bin/env python
import argparse, codecs
from sklearn.externals import joblib
from util import build_features, test_svm, compute_metrics, test_words_in_train, test_words
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NER on SVM')
    parser.add_argument('-d', '--dev', default='dev-raw/', help='dev file')
    parser.add_argument('-s', '--dev_ref', default='dev/', help='reference labels dev file')
    parser.add_argument('-m', '--model', default='models/trained_full_sfx_pfx_0506_2010.pkl', help='Model Trained with SVM')
    opts = parser.parse_args()
    labels_test = []
    predict_test = []
    features_test = []
    clf = joblib.load(opts.model)
    print "Building Features with Dev Set..."
    build_features(opts.dev, opts.dev_ref, features_test, labels_test,1)
    print "Test with Dev Set..."
    test_svm(features_test, predict_test, clf)
    print "Compute Metrics..."
    compute_metrics(predict_test, labels_test)
    with codecs.open('transliterate_words.hi',encoding="utf8",mode="wb") as o_file:
        for ind in predict_test[0]:
            if predict_test[0][ind]==1:
                o_file.write(test_words[ind])
                o_file.write('\n')
    o_file.close()