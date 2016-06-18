# -*- coding: utf-8 -*-

import csv
import numpy
import os
import tensorflow as tf
import sys

from hsv import load

SIZE = 64
LAYER = [32, 64, 1024, 4]

class DataSet(object):
    def __init__(self, images, labels):
        images = numpy.multiply(images.astype(numpy.float32), 1.0 / 255.0)
    
        self._images = images
        self._labels = labels
    
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
  
    @property
    def images(self):
        return self._images
  
    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
  
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
    
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
    
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
  
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def predict(train, test, cache_dir, log_dir):
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, SIZE * SIZE], name = 'x-input')
            x_image = tf.reshape(x, [-1, SIZE, SIZE, 1])

        with tf.name_scope('convolution1'):
            W_conv1 = weight_variable([5, 5, 1, LAYER[0]])
            b_conv1 = bias_variable([LAYER[0]])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            _ = tf.histogram_summary('h_conv1', h_conv1)

        with tf.name_scope('pooling1'):
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('convolution2'):
            W_conv2 = weight_variable([5, 5, LAYER[0], LAYER[1]])
            b_conv2 = bias_variable([LAYER[1]])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pooling2'):
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fully-connected'):
            W_fc1 = weight_variable([(SIZE / 4) * (SIZE / 4) * LAYER[1], LAYER[2]])
            b_fc1 = bias_variable([LAYER[2]])
            h_pool2_flat = tf.reshape(h_pool2, [-1, (SIZE / 4) * (SIZE / 4) * LAYER[1]])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

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
            train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

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
        
                    print 'step %d, training accuracy %g' % (i, acc)
                    saver.save(sess, os.path.join(cache_dir, 'train_data'), global_step = i)
 
            return prediction.eval(feed_dict = {x: test.images, keep_prob: 1.0})


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print '%s [output.csv]' % sys.argv[0]
        sys.exit()

    base_dir = os.path.dirname(sys.argv[0])
    cache_dir = os.path.join(base_dir, 'cache')
    log_dir = os.path.join(base_dir, 'log')

    train_id, train_data, train_label, test_id, test_data = load(base_dir)
    train = DataSet(train_data, dense_to_one_hot(numpy.array(train_label) - 1, LAYER[-1]))
    test = DataSet(test_data, numpy.zeros((test_data.shape[0], LAYER[-1])))

    test_label = predict(train, test, cache_dir, log_dir)

    writer = csv.writer(open(sys.argv[1], 'w'))
    writer.writerow(['id', 'label'])
    writer.writerows(zip(test_id, test_label + 1))
