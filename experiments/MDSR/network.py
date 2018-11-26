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

class MDSR(chainer.Chain):
    """
    MDSR model
    """
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(MDSR, self).__init__()

        with self.init_scope():
            self._forward = []
            self.conv0 = L.Convolution2D(None, 64,
                ksize=9, stride=1, pad=4, initialW=init_w)

            for i in range(80):
                name = 'res_{}'.format(i+1)
                setattr(self, name, ResBlock(64))
                self._forward.append(name)

            self.conv17 = L.Convolution2D(None, 64,
                ksize=3, stride=1, pad=1, initialW=init_w)

            self.conv18_x2 = L.Convolution2D(None, 256,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv19_x2 = L.Convolution2D(None, 3,
                ksize=9, stride=1, pad=4, initialW=init_w)

            self.conv18_x3 = L.Convolution2D(None, 576,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv19_x3 = L.Convolution2D(None, 3,
                ksize=9, stride=1, pad=4, initialW=init_w)

            self.conv18_x4 = L.Convolution2D(None, 256,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv19_x4 = L.Convolution2D(None, 256,
                ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv20_x4 = L.Convolution2D(None, 3,
                ksize=9, stride=1, pad=4, initialW=init_w)

    def enhance(self, x):
        h_skip = self.conv0(x)
        h = h_skip
        for name in self._forward:
            l = getattr(self, name)
            h = l(h)
        h = self.conv17(h) + h_skip
        h_x2 = F.depth2space(self.conv18_x2(h), 2)
        h_x3 = F.depth2space(self.conv18_x3(h), 3)
        h_x4 = F.depth2space(self.conv18_x4(h), 2)
        h_x4 = F.depth2space(self.conv19_x4(h_x4), 2)
        return self.conv19_x2(h_x2), self.conv19_x3(h_x3), self.conv20_x4(h_x4)

    def forward(self, x, t):
        y = self.enhance(x)
        loss = F.mean_absolute_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss
