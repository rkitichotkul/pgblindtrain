'''
Utility function for train/test/val CNN denoisers
'''

import numpy as np
import torch
from torchvision.transforms.functional import rotate
from util.general import calc_psnr, load_checkpoint

dtype = torch.float32

@torch.no_grad()
def batch_psnr(test_image, target_image, max=1.):
    """Calculate average PSNR of a batch of denoised image

    Note:
        The first dimension of the batches must be N (batch size).

    Args:
        test_image (torch.Tensor): batch to calculate PSNR.
        target_image (torch.Tensor): groud truth batch.
        max (float): maximum pixel value on the scale e.g. 1. from [0., 1.].

    Returns:
        psnr (float): average PSNR value.
    """
    psnr = 0
    num_images = test_image.shape[0]
    for i in range(num_images):
        psnr += calc_psnr(test_image[i], target_image[i], max=max)
    psnr /= num_images
    return psnr

def load_checkpoint_train(cpdir, model, optimizer):
    """Load model and optimizer parameters for training

    Note:
        This is simply a wrapper to load_checkpoint so that
        global_step and epoch are updated correctly.
        If cpdir is None, do not load checkpoint and returns
        0 for global_step and epoch.

    Args:
        cpdir (str): path to the checkpoint.
        model: the model to load the parameters to.
        optimizer: the optimizer to load parameters to.

    Returns:
        start_global_step (int): the global step from the checkpoint.
        start_epoch (int): the epoch from the checkpoint.
    """
    start_epoch = 0
    start_global_step = 0
    if cpdir is not None:
        start_global_step, start_epoch = load_checkpoint(
            cpdir, model, optimizer)
        start_global_step += 1
        start_epoch += 1
    return start_global_step, start_epoch

def log_hyperparams(batch_size, sigma, alpha, writer):
    """Log training hyperparameters

    Args:
        batch_size (int): batch size.
        sigma (list): standard deviation of Gaussian noise.
        alpha (list): Poisson noise strength
        writer: tensorboard.SummaryWriter object.
    """
    min_sigma = sigma[0].item()
    if len(sigma) == 1:
        max_sigma = -1
    else:
        max_sigma = sigma[1].item()
    min_alpha = alpha[0].item()
    if len(alpha) == 1:
        max_alpha = -1
    else:
        max_alpha = alpha[1].item()
    hparam_dict = {
        'batch_size': batch_size,
        'min_sigma': min_sigma,
        'max_sigma': max_sigma,
        'min_alpha': min_alpha,
        'max_alpha': max_alpha
    }
    writer.add_hparams(hparam_dict, {})

def get_val_params(sigma, alpha):
    """Get sigma and alpha for validation

    Note:
        If the length of a parameter is 1, returns itself.
        If the length is 2, returns the average.

    Args:
        sigma (list): training sigma argument.
        alpha (list): training alpha argument

    Returns:
        sigma_val (list): validation sigma.
        alpha_val (list): validation alpha

    Raise:
        Error if the length if an argument is not 1 or 2.
    """
    if len(sigma) == 2:
        sigma_val = [(sigma[0] + sigma[1]) / 2]
    elif len(sigma) == 1:
        sigma_val = sigma
    else:
        raise RuntimeError('Invalid sigma length = {}'.format(len(sigma)))

    if len(alpha) == 2:
        alpha_val = [(alpha[0] + alpha[1]) / 2]
    elif len(alpha) == 1:
        alpha_val = alpha
    else:
        raise RuntimeError('Invalid alpha length = {}'.format(len(alpha)))

    return sigma_val, alpha_val

class FixedAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = float(np.random.choice(self.angles))
        return rotate(x, angle)
