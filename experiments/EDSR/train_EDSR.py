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

from network import EDSR
from dataset import DIV2KDataset

# このファイルの絶対パス
FILE_PATH = Path(__file__).resolve().parent
ROOT_PATH = FILE_PATH.parent.parent
RESULT_PATH = ROOT_PATH.joinpath('results/EDSR')
MODEL_PATH = ROOT_PATH.joinpath('models/EDSR')


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
    args = parser.parse_args()

    print('### Learning Parameter ###')
    print('# Dataset: {}'.format(args.dataset))
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    out_path = RESULT_PATH.joinpath('EDSR')
    print("# result dir : {}".format(out_path))
    out_path.mkdir(parents=True, exist_ok=True)

    model = EDSR

    train = DIV2KDataset(scale=2, size=64, dataset='train')
    test = DIV2KDataset(scale=2, size=64, dataset='valid')
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
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_path)

    #Set up Updater Extentions
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.ExponentialShift("alpha", 0.5), trigger=(100, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, filename='model_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(extensions.ProgressBar())

    # start training
    trainer.run()

if __name__ == '__main__':
    main()
