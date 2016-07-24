# -*- coding: utf-8 -*-

import argparse
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import cv2
import pandas as pd
#from matplotlib import pylab as plt

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=10,
                        help='Number of units')
    args = parser.parse_args()

    # load csv
    n_in = 32*32

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Load model
    model = L.Classifier(MLP(n_in, args.unit, 3))
    serializers.load_npz('model/simple-3layer-perceptron.model', model)

    # Load optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    serializers.load_npz('model/simple-3layer-perceptron.state', optimizer)

    # Load dataset from CSV
    csv = pd.read_csv('csv/images-data.csv')

    dd = []

    for file, label in zip(csv['file'], csv['label']):

        print file, label

        # load a color image
        img = cv2.imread(file, cv2.IMREAD_COLOR)

        # color -> grayscale
        imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # image -> array
        imgdata = np.array(imggray, dtype='f')

        # set dataset
        x = imgdata.reshape(1, -1)[0]
        x = x / 255.0
        y = np.array(label, dtype=np.int32)
        dataset = (x, y)

        dd.append(dataset)

    # Predictor
    xx = Variable(np.array([dd[1][0],]), volatile=True)
    y = model.predictor(xx)
    print y.data
    print np.argmax(y.data)

if __name__ == '__main__':
    main()