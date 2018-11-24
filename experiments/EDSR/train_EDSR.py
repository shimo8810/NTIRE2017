"""
"""
from pathlib import Path
import argparse
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import training
from chainer.training import extensions

from network import Baseline
from dataset import DIV2KDataset

def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=200, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--dataset', '-d', type=str, choices=['coil', 'mmnist'], default='mmnist',
                        help='using dataset')
    # Hyper Parameter
    parser.add_argument('--latent', '-l', default=100, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--coef1', type=float, default=1.0,
                        help='')
    parser.add_argument('--coef2', type=float, default=0.5,
                        help='')
    parser.add_argument('--coef3', type=float, default=1.0,
                        help='')
    parser.add_argument('--coef4', type=float, default=0.01,
                        help='')
    parser.add_argument('--ch', type=int, default=4,
                        help='')
    args = parser.parse_args()

    print('### Learning Parameter ###')
    print('# Dataset: {}'.format(args.dataset))
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = Baseline()

    train = DIV2KDataset(scale=2, size=48, dataset='train')
    test = DIV2KDataset(scale=2, size=48, dataset='valid')
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                    repeat=False, shuffle=False)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=10e-4)
    optimizer.setup(model)

    # Set up an updater
    updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=0)

    # Set up a Trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    #Set up Updater Extentions
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.ExponentialShift("alpha", 0.5), trigger=(100, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, filename='model_epoch-{.updater.epoch}'), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    # start training
    trainer.run()

if __name__ == '__main__':
    main()