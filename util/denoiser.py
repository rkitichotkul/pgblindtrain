"""
Collection of denoiser wrappers suitable for passing to MSE/SURE heatmap generation functions.

TODO: I changed how DnCNN works. It now outputs the denoised image instead of the predicted noise.
Changes must be made here accordingly.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
from bm3d import bm3d
from train.model import DnCNN, Remez
from util import general as gutil

class CNN_denoiser:
    def __init__(self, model, batch_size=None, device=torch.device('cpu')):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def __call__(self, image, std=None):
        image = image.expand(1, -1, -1, -1).to(device=self.device, dtype=torch.float32)
        output = self.model(image)
        output = output.squeeze(dim=0)
        return output.cpu()

class BM3D_denoiser:
    def __init__(self, std=None):
        self.std = std

    def __call__(self, image, std=None):
        if std is None:
            std = self.std
        return torch.Tensor(bm3d(image[0], std)).unsqueeze(0)

    def set_std(self, std):
        self.std = std

def setup_DnCNN(modedir, num_layers=17, device=torch.device('cpu')):
    model = DnCNN(num_layers=num_layers)
    gutil.load_checkpoint(modedir, model, None, device=device)
    model.to(device=device)
    model.eval()
    return model

def setup_Remez(modedir, num_layers=20, device=torch.device('cpu')):
    model = Remez(num_layers=num_layers)
    gutil.load_checkpoint(modedir, model, None, device=device)
    model.to(device=device)
    model.eval()
    return model
