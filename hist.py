# -*- coding: utf-8 -*-

import csv
import cv2
import numpy
import os
import pickle
import sys
import xgboost

import sklearn.cross_validation
import sklearn.ensemble
import sklearn.svm

SIZE = 64

def vectorize(image_dir, files, cache_path = None):
    try:
        return pickle.load(open(cache_path))
    except:
        pass

    data = None

    for filename in files:
        print filename

        im = cv2.imread(os.path.join(image_dir, filename + '.jpg'))
        v = numpy.vstack(list(im.shape[:2]) + [cv2.calcHist(im, [i], None, [SIZE], [0, 255]) for i in range(im.shape[2])]).flatten()

        if data == None:
            data = v
        else:
            data = numpy.vstack([data, v])

    if cache_path:
        pickle.dump(data, open(cache_path, 'w'))

    return data

def load(base_dir):
    image_dir = os.path.join(base_dir, 'roof_images')
    cache_path = lambda name: os.path.join(base_dir, 'cache', name)
    read_csv = lambda name: [line for line in csv.reader(open(os.path.join(base_dir, name)))]

    train_files = read_csv('id_train.csv')[1:]
    train_id = [file[0] for file in train_files]
    train_data = vectorize(image_dir, train_id, cache_path('hist_train_' + str(SIZE)))
    train_label = map(int, [file[1] for file in train_files])

    test_files = read_csv('sample_submission4.csv')[1:]
    test_id = [file[0] for file in test_files]
    test_data = vectorize(image_dir, test_id, cache_path('hist_test_' + str(SIZE)))

    return train_id, train_data, train_label, test_id, test_data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print '%s [output.csv]' % sys.argv[0]
        sys.exit()

    train_id, train_data, train_label, test_id, test_data = load(os.path.dirname(sys.argv[0]))

    # 32 / Accuracy: 0.434875 (+/- 0.000423)
    # 64 / Accuracy: 0.434875 (+/- 0.000423) 
    # classifier = sklearn.svm.SVC()

    # 32 / Accuracy: 0.375563 (+/- 0.241356) 
    # 64 / Accuracy: 0.378996 (+/- 0.221394)
    # classifier = sklearn.svm.LinearSVC()

    # 32 / Accuracy: 0.495126 (+/- 0.028564)
    # 64 / Accuracy: 0.486746 (+/- 0.028496)
    # classifier = sklearn.ensemble.RandomForestClassifier()

    # 32 / Accuracy: 0.552248 (+/- 0.034961)
    # 64 / Accuracy: 0.551620 (+/- 0.033765)
    # classifier = sklearn.ensemble.AdaBoostClassifier()

    # 32 / Accuracy: 0.563994 (+/- 0.034092)
    # 64 / Accuracy: 0.562369 (+/- 0.044722)
    classifier = xgboost.XGBClassifier()

    scores = sklearn.cross_validation.cross_val_score(classifier, train_data, train_label, cv = 5, n_jobs = 5)

    print scores
    print 'Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 2)

    classifier.fit(train_data, train_label)
    test_label = classifier.predict(test_data)

    writer = csv.writer(open(sys.argv[1], 'w'))
    writer.writerow(['id', 'label'])
    writer.writerows(zip(test_id, test_label))
