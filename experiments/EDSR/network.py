import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class ResBlock(chainer.Chain):
    """
    conv -> relu -> conv
    """
    def __init__(self, ch_size=64, scale=None):
        self.scale = scale
        init_w = chainer.initializers.HeNormal()
        super(ResBlock, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch_size,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv2 = L.Convolution2D(None, ch_size,
                ksize=3, stride=1, pad=1, initialW=init_w)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        if self.scale:
            h = x * self.scale + self.conv2(h)
        else:
            h = x + self.conv2(h)
        return h

class Baseline(chainer.Chain):
    """
    balse line model
    """
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        self._forward = []
        super(Baseline, self).__init__()

        with self.init_scope():
            self.conv0 = L.Convolution2D(None, 64,
                ksize=9, stride=1, pad=4, initialW=init_w)

            for i in range(16):
                name = 'res_{}'.format(i+1)
                setattr(self, name, ResBlock(64))
                self._forward.append(name)

            self.conv17 = L.Convolution2D(None, 64,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv18 = L.Convolution2D(None, 256,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv19 = L.Convolution2D(None, 3,
                ksize=9, stride=1, pad=4, initialW=init_w)

    def enhance(self, x):
        h_skip = self.conv0(x)
        h = h_skip
        for name in self._forward:
            l = getattr(self, name)
            h = l(h)
        h = self.conv17(h) + h_skip
        h = F.depth2space(self.conv18(h), 2)
        return self.conv19(h)

    def forward(self, x, t):
        y = self.enhance(x)
        loss = F.mean_absolute_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss

class EDSR(chainer.Chain):
    """
    EDSR model
    """
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(EDSR, self).__init__()

        with self.init_scope():
            self._forward = []
            self.conv0 = L.Convolution2D(None, 64,
                ksize=9, stride=1, pad=4, initialW=init_w)

            for i in range(32):
                name = 'res_{}'.format(i+1)
                setattr(self, name, ResBlock(256, scale=0.1))
                self._forward.append(name)

            self.conv17 = L.Convolution2D(None, 64,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv18 = L.Convolution2D(None, 256,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv19 = L.Convolution2D(None, 3,
                ksize=9, stride=1, pad=4, initialW=init_w)

    def enhance(self, x):
        h_skip = self.conv0(x)
        h = h_skip
        for name in self._forward:
            l = getattr(self, name)
            h = l(h)
        h = self.conv17(h) + h_skip
        h = F.depth2space(self.conv18(h), 2)
        return self.conv19(h)

    def forward(self, x, t):
        y = self.enhance(x)
        loss = F.mean_absolute_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss
