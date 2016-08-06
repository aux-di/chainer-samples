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

        n = 0

        if n == 0:
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
        elif n == 1:
            h1 = F.clipped_relu(self.l1(x))
            h2 = F.clipped_relu(self.l2(h1))
        elif n == 2:
            h1 = F.crelu(self.l1(x))
            h2 = F.crelu(self.l2(h1))
        elif n == 3:
            h1 = F.elu(self.l1(x))
            h2 = F.elu(self.l2(h1))
        elif n == 4:
            h1 = F.hard_sigmoid(self.l1(x))
            h2 = F.hard_sigmoid(self.l2(h1))
        elif n == 5:
            h1 = F.leaky_relu(self.l1(x))
            h2 = F.leaky_relu(self.l2(h1))
        elif n == 6:
            h1 = F.log_softmax(self.l1(x))
            h2 = F.log_softmax(self.l2(h1))
        elif n == 7:
            h1 = F.sigmoid(self.l1(x))
            h2 = F.sigmoid(self.l2(h1))
        elif n == 8:
            h1 = F.softmax(self.l1(x))
            h2 = F.softmax(self.l2(h1))
        elif n == 9:
            h1 = F.softplus(self.l1(x))
            h2 = F.softplus(self.l2(h1))
        elif n == 10:
            h1 = F.tanh(self.l1(x))
            h2 = F.tanh(self.l2(h1))

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

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    n_in = 32*32

    model = L.Classifier(MLP(n_in, args.unit, 3))

    # Setup an optimizer
    n = 8
    if n == 0:
        optimizer = chainer.optimizers.Adam()
    elif n == 1:
        optimizer = chainer.optimizers.AdaDelta()
    elif n == 2:
        optimizer = chainer.optimizers.AdaGrad()
    elif n == 3:
        optimizer = chainer.optimizers.MomentumSGD()
    elif n == 4:
        optimizer = chainer.optimizers.NesterovAG()
    elif n == 5:
        optimizer = chainer.optimizers.RMSprop()
    elif n == 6:
        optimizer = chainer.optimizers.RMSpropGraves()
    elif n == 7:
        optimizer = chainer.optimizers.SGD()
    elif n == 8:
        optimizer = chainer.optimizers.SMORMS3()

    optimizer.setup(model)

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

        gray = []

        for y in range(len(imggray)):
            for x in range(len(imggray[y])):
                gray.append(imggray[y][x])

        imgdata = np.array(gray, dtype='f').reshape(1, 1, 32, 32) / 255.0

        # set dataset
        x = imgdata
        y = np.array(label, dtype=np.int32)
        dataset = (x, y)

        dd.append(dataset)

    train, test = dd, dd

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot())

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Resume from a snapshot
    #chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()

    # Predictor
    xx = Variable(np.array([dd[1][0], ]), volatile=True)
    y = model.predictor(xx)
    print y.data
    print np.argmax(y.data)

if __name__ == '__main__':
    main()