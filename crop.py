# -*- coding: utf-8 -*-

import csv
import cv2
import os
import sys

SIZE = 64
STRIDE = 4
WIDTH = 16 # 0 for test data

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print '%s [id_train.csv] [input directory] [output directory]' % sys.argv[0]
        sys.exit()

    files = [line for line in csv.reader(open(sys.argv[1]))][1:]
    for filename, label in files:
        print filename

        im = cv2.imread(os.path.join(sys.argv[2], '%s.jpg' % filename))
        aspect = float(im.shape[0]) / im.shape[1]

        ratio = SIZE / float(min(im.shape[0], im.shape[1]))
        im = cv2.resize(im, tuple(map(lambda x: int(round(x * ratio)), im.shape[1::-1])), interpolation = cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
        im, _, _ = cv2.split(im)

        if aspect <= 0.8:
            resolution = (SIZE, int(SIZE / 0.8))
        elif 0.8 < aspect < 1.25:
            resolution = (SIZE, SIZE)
        elif 1.25 <= aspect:
            resolution = (int(SIZE / 0.8), SIZE)

        margin = map(lambda (x, y): (x - y) / 2, zip(im.shape, resolution))
        cnt = 0

        for i in range(max(0, margin[0] - WIDTH), min(margin[0] * 2 + 1, margin[0] + WIDTH + 1), STRIDE):
            for j in range(max(0, margin[1] - WIDTH), min(margin[1] * 2 + 1, margin[1] + WIDTH + 1), STRIDE):
                cv2.imwrite(os.path.join(sys.argv[3], label, '%s_%04d.jpg' % (filename, cnt)), im[i: i + resolution[0], j: j + resolution[1]])
                cnt += 1
