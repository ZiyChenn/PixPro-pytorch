import os
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def is_image_file(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_imgs(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


class PairwiseView(Dataset):
    def __init__(
        self,
        root,
        space_trans,
        color_trans,
        color_trans2=None
    ):
        super(PairwiseView, self).__init__()
        self.source = find_imgs(root)
        self.space_trans = space_trans
        self.color_trans = color_trans
        if color_trans2 is None:
            self.color_trans2 = color_trans
        else:
            self.color_trans2 = color_trans2

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        fpath = self.source[idx]
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        y = np.arange(0, image.shape[0], dtype=np.float32)
        x = np.arange(0, image.shape[1], dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x)
        grid_y, grid_x = grid_y.T, grid_x.T

        view1_ = self.space_trans(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])
        view2_ = self.space_trans(image=image, grid_y=grid_y[..., None], grid_x=grid_x[..., None])

        view1 = view1_['image']
        view1 = self.color_trans(image=view1)['image']
        grid1 = np.concatenate([view1_['grid_y'], view1_['grid_x']], axis=-1)
        grid1 = torch.from_numpy(grid1).permute(2, 0, 1)  # (H, W, C) --> (C, H, W)

        view2 = view2_['image']
        view2 = self.color_trans2(image=view2)['image']
        grid2 = np.concatenate([view2_['grid_y'], view2_['grid_x']], axis=-1)
        grid2 = torch.from_numpy(grid2).permute(2, 0, 1)

        H1 = grid1[0][0][0] - grid1[0][-1][-1]
        W1 = grid1[1][0][0] - grid1[1][-1][-1]
        L1 = torch.sqrt(H1 ** 2 + W1 ** 2)

        H2 = grid2[0][0][0] - grid2[0][-1][-1]
        W2 = grid2[1][0][0] - grid2[1][-1][-1]
        L2 = torch.sqrt(H2 ** 2 + W2 ** 2)

        size = torch.max(L1, L2)

        output = {
            'size': size,
            'view1': view1,
            'grid1': grid1,
            'view2': view2,
            'grid2': grid2,
        }

        return output
