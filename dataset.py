# -*- coding: utf-8 -*-

import numpy

class DataSet(object):
    def __init__(self, images, labels):
        images = numpy.multiply(images.astype(numpy.float32), 1.0 / 255.0)
        if len(labels.shape) == 1:
            labels = self.dense_to_one_hot(labels, labels.max() + 1)
    
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

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
