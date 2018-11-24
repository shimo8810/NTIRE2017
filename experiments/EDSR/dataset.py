from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
from chainer.dataset import dataset_mixin
from chainercv import transforms

class DIV2KDataset(dataset_mixin.DatasetMixin):
    def __init__(self, scale=4, size=64, dataset='valid'):
        assert scale in [2, 3, 4], "scale parameter must be 2, 3 or 4."
        assert dataset in ['train', 'valid'], "scale parameter must be 2, 3 or 4."

        self.scale = scale
        self.size = size
        lr_paths = sorted(Path('DIV2K_{}_LR_bicubic/X{}'.format(dataset, scale)).iterdir())
        hr_paths = sorted(Path('DIV2K_{}_HR'.format(dataset)).iterdir())
        assert len(lr_paths) == len(hr_paths), "the number of HR images must be same as the number of LR images."
        self.data = []
        c = 0
        for lr_p, hr_p in tqdm(zip(lr_paths, hr_paths)):
            c += 1
            # load images
            image_lr = np.array(Image.open(lr_p)) \
                .astype(np.float32).transpose(2, 0, 1) / 255.
            image_hr = np.array(Image.open(hr_p)) \
                .astype(np.float32).transpose(2, 0, 1) / 255.

            self.data.append([image_lr, image_hr])

            # すべてのデータセットを読み込めないので
            if dataset == 'valid' and c >= 50:
                break
            if dataset == 'train' and c >= 200:
                break

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        # crop images
        image_lr, image_hr = self.data[i]
        image_lr, sl = transforms.random_crop(image_lr,
                            (self.size, self.size), return_param=True)
        image_hr = image_hr[:, sl['y_slice'].start*self.scale:sl['y_slice'].stop*self.scale,
                    sl['x_slice'].start*self.scale: sl['x_slice'].stop*self.scale]
        return image_lr, image_hr