"""Implement the Dataset class for CNN denoiser train/eval/test.

Extends torch.utils.data.Dataset class for training and testing.

Example
-------
    To create a torch.utils.data.Dataloader
    from a DenoiseDataset dataset, use::

        data_loader = DataLoader(dataset).

Note
----
    A .h5 file storing the dataset must be formatted as
        dataset.h5
            - key 1 : image 1
            - key 2 : image 2
            - ...
        where each image has dimensions (C, H, W).
"""

import torch
from torch.utils.data import Dataset
from numpy.random import uniform
import h5py

class DenoiseDataset(Dataset):
    def __init__(self, datadir, sigma, alpha, transforms=None):
        """
        Arguments
        ---------
            datadir (str): path to .h5 file.
            sigma (list): Gaussian noise standard deviation.
            alpha (list): the Poisson noise strength.
            transform: image transformation, for data augmentation.

        Note
        ----
            If sigma or alpha are lists of two entries, the value used will be
            uniformly sampled from the two values.
        """
        super(Dataset, self).__init__()
        if (len(sigma) not in {1, 2}) or (len(alpha) not in {1, 2}):
            print('sigma', sigma)
            print('alpha', alpha)
            raise ValueError('Invalid format of sigma or alpha to DenoiseDataset')
        self.datadir = datadir
        self.sigma = sigma
        self.alpha = alpha
        self.transforms = transforms
        h5f = h5py.File(datadir, 'r')
        self.keys = list(h5f.keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """Return image, noise pair of the same size (C, H, W)."""
        # Load image
        h5f = h5py.File(self.datadir, 'r')
        key = self.keys[idx]
        image = torch.Tensor(h5f[key])
        h5f.close()
        if self.transforms is not None:
            image = self.transforms(image)

        # Generate noisy image
        if len(self.sigma) == 1:
            sigma = self.sigma[0]
        else:
            sigma = uniform(self.sigma[0], self.sigma[1])
        if len(self.alpha) == 1:
            alpha = self.alpha[0]
        else:
            alpha = uniform(self.alpha[0], self.alpha[1])

        if alpha != 0:
            noisy_image = alpha * torch.poisson(image / alpha)
        else:
            noisy_image = image.clone()
        if std != 0:
            noisy_image += torch.normal(mean=torch.zeros_like(image), std=std)

        return image, noisy_image
