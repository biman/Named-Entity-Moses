#!/usr/bin/env python
from sklearn import svm
import argparse, os, pickle
import codecs,time
import datetime
from sklearn.externals import joblib
from util import build_features, train_svm
clf = svm.SVC()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NER on SVM')
    parser.add_argument('-i', '--input', default='training-hindi-raw/', help='input file')
    parser.add_argument('-r', '--reference', default='training-hindi/', help='reference labels train file')
    opts = parser.parse_args()
    features_train = []
    labels_train = []
    print "Building Features with Train Set..."
    build_features(opts.input, opts.reference, features_train, labels_train,0)
    print "Start Training..."
    train_svm(features_train,labels_train)