"""
Collection of denoiser wrappers suitable for passing to MSE/SURE heatmap generation functions.

TODO: I changed how DnCNN works. It now outputs the denoised image instead of the predicted noise.
Changes must be made here accordingly.
"""

import os
import numpy as np
import torch
from torch import nn
from bm3d import bm3d
from util import general as gutil

class DnCNN_denoiser:
    def __init__(self, model, batch_size=None, device=torch.device('cpu')):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def __call__(self, image, std=None):
        """std is there only to fit the template (see csalgo.py). we don't do std here."""
        if len(image.shape) == 3:
            image = image.expand(1, -1, -1, -1).to(device=self.device, dtype=torch.float32)
            output = image - self.model(image)
            output = output.squeeze(dim=0)
        elif len(image.shape) == 4:
            image = image.to(device=self.device, dtype=torch.float32)
            if self.batch_size is None:
                output = image - self.model(image)
            else:
                output = torch.zeros(*image.shape)
                num_batches = int(ceil(image.shape[0] / self.batch_size))
                for i in range(num_batches):
                    this_batch = image[i * self.batch_size : (i + 1) * self.batch_size]
                    output[i * self.batch_size : (i + 1) * self.batch_size] = this_batch - self.model(this_batch)
        else:
            raise Error('Image shape is not 3 (CHW) or 4 (NCHW).')
        return output.cpu()

class BM3D_denoiser:
    def __init__(self, std=None, isClamp=False):
        self.std = std
        self.isClamp = isClamp

    def __call__(self, image, std=None):
        if self.isClamp:
            image = gutil.clamp(image)
        if std is None:
            std = self.std
        return torch.Tensor(bm3d(image[0], std)).unsqueeze(0)

    def set_std(self, std):
        self.std = std


class DnCNN_ensemble_denoiser:
    def __init__(self, models, std_ranges, device=torch.device('cpu'), verbose=False, std=None):
        self.models = models
        self.std_ranges = std_ranges    # Need to be np.ndarray
        self.device = device
        self.verbose = verbose
        self.std = std

    def __call__(self, image, std=None):
        if std is None:
            std = self.std

        select = np.sum(std > self.std_ranges) - 1
        if select < 0:
            print('denoiser.DnCNN_ensemble_denoiser: The noise level is lower than models available')
            select += 1
        elif select > len(self.models) - 1:
            print('denoiser.DnCNN_ensemble_denoiser: The noise level is higher than models available')
            select -= 1
        if self.verbose:
            print('denoiser.DnCNN_ensemble_denoiser: select = {:d}'.format(select))
        image = image.expand(1, -1, -1, -1).to(device=self.device, dtype=torch.float32)
        output = image - self.models[select](image)
        return output.squeeze(dim=0).cpu()

def setup_DnCNN(modedir, num_layers=17, device=torch.device('cpu')):
    model = DnCNN(1, num_layers=num_layers)
    gutil.load_checkpoint(modedir, model, None, device=device)
    model.to(device=device)
    model.eval()
    return model

def setup_DnCNN_ensemble(path, modelnames, num_layers=20, device=torch.device('cpu')):
    models = [None] * len(modelnames)
    for i, name in enumerate(modelnames):
        models[i] = setup_DnCNN(os.path.join(path, '{}.pth'.format(name)),
                                num_layers=num_layers, device=device)
    return models

class DnCNN(nn.Module):
    def __init__(self, channels, num_layers=17):
        super(DnCNN, self).__init__()

        # Fixed parameters
        kernel_size = 3
        padding = 1
        features = 64

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.layers(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
