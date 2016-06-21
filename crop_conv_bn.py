# -*- coding: utf-8 -*-

import csv
import cv2
import glob
import numpy
import os
import pickle
import tensorflow as tf
import sys

from dataset import DataSet

RESOLUTION = [(64, 80), (64, 64), (80, 64)]
LAYER = [32, 64, 1024, 4]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def batch_normalization(shape, input):
    eps = 1e-5
    gamma = weight_variable([shape])
    beta = weight_variable([shape])
    mean, variance = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def predict(train, test, shape, cache_dir, log_dir, testing = False):
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, shape[0] * shape[1]], name = 'x-input')
            x_image = tf.reshape(x, [-1, shape[0], shape[1], 1])

        with tf.name_scope('convolution1'):
            W_conv1 = weight_variable([5, 5, 1, LAYER[0]])
            h_conv1 = conv2d(x_image, W_conv1)
            bn1 = batch_normalization(LAYER[0], h_conv1)
            h_pool1 = max_pool_2x2(tf.nn.relu(bn1))
            _ = tf.histogram_summary('h_conv1', h_conv1)

        with tf.name_scope('pooling1'):
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('convolution2'):
            W_conv2 = weight_variable([5, 5, LAYER[0], LAYER[1]])
            h_conv2 = conv2d(h_pool1, W_conv2)
            bn2 = batch_normalization(LAYER[1], h_conv2)
            h_pool2 = max_pool_2x2(tf.nn.relu(bn2))

        with tf.name_scope('pooling2'):
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fully-connected'):
            W_fc1 = weight_variable([(shape[0] / 4) * (shape[1] / 4) * LAYER[1], LAYER[2]])
            h_pool2_flat = tf.reshape(h_pool2, [-1, (shape[0] / 4) * (shape[1] / 4) * LAYER[1]])
            bn3 = batch_normalization(LAYER[2], tf.matmul(h_pool2_flat, W_fc1))
            h_fc1 = tf.nn.relu(bn3)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('readout'):
            W_fc2 = weight_variable([LAYER[2], LAYER[3]])
            b_fc2 = bias_variable([LAYER[3]])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope('optimizer'):
            y_ = tf.placeholder(tf.float32, [None, LAYER[3]], name = 'y-input')
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
            train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

            prediction = tf.argmax(y_conv, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            writer = tf.train.SummaryWriter(log_dir, sess.graph)
            sess.run(tf.initialize_all_variables())
    
            for i in range(5001):
                batch = train.next_batch(50)
                train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
        
                if i % 100 == 0:
                    summary_str, acc = sess.run([tf.merge_all_summaries(), accuracy], feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
                    writer.add_summary(summary_str, i)
        
                    print 'step %d, training accuracy %f' % (i, acc)
                    saver.save(sess, os.path.join(cache_dir, 'train_data'), global_step = i)

            if testing:
                acc = accuracy.eval(feed_dict = {x: test.images, y_: test.labels, keep_prob: 1.0})
                print 'test accuracy %f' % acc
            else:
                return prediction.eval(feed_dict = {x: test.images, keep_prob: 1.0})

def vectorize(image_dir, files, cache_path = None, test = False):
    try:
        return pickle.load(open(cache_path))
    except:
        pass

    ids = [[], [], []]
    data = [None, None, None]
    labels = [[], [], []]

    for name, label in files:
        print name

        if test:
            crops = glob.glob(os.path.join(image_dir, 'test', '%s.jpg' % name))
        else:
            crops = glob.glob(os.path.join(image_dir, str(label), '%s_*.jpg' % name))

        for filename in crops:
            im = cv2.imread(filename)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            if im.shape[0] < im.shape[1]:
                target = 0
            elif im.shape[0] == im.shape[1]:
                target = 1
            elif im.shape[0] > im.shape[1]:
                target = 2

            v = im.flatten()
            # Global Contrast Normalization
            v = (v - v.mean()) / v.std()

            if data[target] == None:
                data[target] = v
            else:
                data[target] = numpy.vstack([data[target], v])

            ids[target].append(os.path.basename(filename)[:-4])
            labels[target].append(int(label))

    # ZCA Whitening
    for i in range(3):
        eps = 1e-5
        sigma = numpy.dot(data[i], data[i].T) / float(data[i].shape[0])
        U, S, V = numpy.linalg.svd(sigma)
        zca = numpy.dot(numpy.dot(U, numpy.diag(1.0 / numpy.sqrt(S + eps))), U.T)
        data[i] = numpy.dot(zca, data[i])

    ids = [numpy.array(id) for id in ids]
    labels = [numpy.array(label) - 1 for label in labels]
    result = (ids, data, labels)

    if cache_path:
        pickle.dump(result, open(cache_path, 'w'))

    return result

def load(base_dir):
    image_dir = os.path.join(base_dir, 'roof_images_cropped')
    cache_path = lambda name: os.path.join(base_dir, 'cache', name)
    read_csv = lambda name: [line for line in csv.reader(open(os.path.join(base_dir, name)))]

    train_files = read_csv('id_train.csv')[1:]
    train_id, train_data, train_label = vectorize(image_dir, train_files, cache_path('crop_train'))

    test_files = read_csv('sample_submission4.csv')[1:]
    test_id, test_data, test_label = vectorize(image_dir, test_files, cache_path('crop_test'), True)

    return train_id, train_data, train_label, test_id, test_data, test_label

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print '%s [output.csv] / %s [k-fold] [index]' % sys.argv[0]
        sys.exit()

    base_dir = os.path.dirname(sys.argv[0])
    cache_dir = os.path.join(base_dir, 'cache')
    log_dir = os.path.join(base_dir, 'log')

    train_id, train_data, train_label, test_id, test_data, test_label = load(base_dir)

    if len(sys.argv) == 3:
        for i in range(3):
            N = train_id[i].shape[0]
            K, index = map(int, sys.argv[1:])
    
            train_index = range(N * index / K) + range(N * (index + 1) / K, N)
            test_index = range(N * index / K, N * (index + 1) / K)
    
            test_id[i], test_data[i], test_label[i] = train_id[i][test_index], train_data[i][test_index], train_label[i][test_index]
            train_id[i], train_data[i], train_label[i] = train_id[i][train_index], train_data[i][train_index], train_label[i][train_index]

    if len(sys.argv) == 2:
        writer = csv.writer(open(sys.argv[1], 'w'))
        writer.writerow(['id', 'label'])

    for i in range(3):
        train = DataSet(train_data[i], train_label[i])
        test = DataSet(test_data[i], test_label[i])

        predict_label = predict(train, test, RESOLUTION[i], cache_dir, log_dir, len(sys.argv) == 3)

        if len(sys.argv) == 2:
            writer.writerows(zip(test_id[i], predict_label + 1))
