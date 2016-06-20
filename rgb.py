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
        im = cv2.resize(im, (SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
        im = numpy.vstack(cv2.split(im))

        v = im.reshape(SIZE * SIZE * 3)
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
    train_data = vectorize(image_dir, train_id, cache_path('rgb_train_' + str(SIZE)))
    train_label = map(int, [file[1] for file in train_files])

    test_files = read_csv('sample_submission4.csv')[1:]
    test_id = [file[0] for file in test_files]
    test_data = vectorize(image_dir, test_id, cache_path('rgb_test_' + str(SIZE)))

    return train_id, train_data, train_label, test_id, test_data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print '%s [output.csv]' % sys.argv[0]
        sys.exit()

    train_id, train_data, train_label, test_id, test_data = load(os.path.dirname(sys.argv[0]))

    # Accuracy: 0.434875 (+/- 0.000423)
    # classifier = sklearn.svm.SVC()

    # Accuracy: 0.478015 (+/- 0.038127)
    # classifier = sklearn.svm.LinearSVC()

    # Accuracy: 0.609872 (+/- 0.027341)
    # classifier = sklearn.ensemble.RandomForestClassifier()

    # Accuracy: 0.574123 (+/- 0.039928)
    # classifier = sklearn.ensemble.AdaBoostClassifier()

    # Accuracy: 0.659373 (+/- 0.035545)
    classifier = xgboost.XGBClassifier()

    scores = sklearn.cross_validation.cross_val_score(classifier, train_data, train_label, cv = 5, n_jobs = 5)

    print scores
    print 'Accuracy: %f (+/- %f)' % (scores.mean(), scores.std() * 2)

    classifier.fit(train_data, train_label)
    test_label = classifier.predict(test_data)

    writer = csv.writer(open(sys.argv[1], 'w'))
    writer.writerow(['id', 'label'])
    writer.writerows(zip(test_id, test_label))
