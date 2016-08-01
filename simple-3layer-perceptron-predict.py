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
    parser.add_argument('--epoch', '-e', type=int, default=100,
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

    # load a color image
    img1 = cv2.imread('images/zero.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('images/black.jpg', cv2.IMREAD_COLOR)
    img3 = cv2.imread('images/white.jpg', cv2.IMREAD_COLOR)

    # color -> grayscale
    imggray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    imggray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    imggray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

    # image -> array
    gray = []

    for y in range(len(imggray1)):
        for x in range(len(imggray1[y])):
            gray.append(imggray1[y][x])

    imgdata1 = np.array(gray, dtype='f')
    imgdata1 = imgdata1.reshape(1, 1, 32, 32)
    imgdata1 = imgdata1 / 255.0

    gray = []

    for y in range(len(imggray2)):
        for x in range(len(imggray2[y])):
            gray.append(imggray2[y][x])

    imgdata2 = np.array(gray, dtype='f')
    imgdata2 = imgdata2.reshape(1, 1, 32, 32)
    imgdata2 = imgdata2 / 255.0

    gray = []

    for y in range(len(imggray3)):
        for x in range(len(imggray3[y])):
            gray.append(imggray3[y][x])

    imgdata3 = np.array(gray, dtype='f')
    imgdata3 = imgdata3.reshape(1, 1, 32, 32)
    imgdata3 = imgdata3 / 255.0

    n_in = 32*32

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(MLP(n_in, args.unit, 3))

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load dataset
    x1 = imgdata1
    x2 = imgdata2
    x3 = imgdata3
    y1 = np.array(0, dtype=np.int32)
    y2 = np.array(1, dtype=np.int32)
    y3 = np.array(2, dtype=np.int32)
    dd = [(x1, y1), (x2, y2), (x3, y3)]
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
    xx = Variable(np.array([x1, ]), volatile=True)
    y = model.predictor(xx)
    print y.data
    print np.argmax(y.data)

if __name__ == '__main__':
    main()