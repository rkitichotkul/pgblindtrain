"""
General utility functions

Tensor manipulation functions
* to_tensor
* to_PIL
* normalize
* clamp

Noise-related functions
* calc_psnr
* generate_noise
* add_noise

File/Directory manipulation functions
* generate_loadlist
* generate_namelist
* remove_if_exists

Neural-network-related functions
* load_checkpoint

"""

import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.functional import mse_loss
from PIL import Image

"""Transform PIL image to torch.Tensor (C, H, W)"""
to_tensor = transforms.ToTensor()

"""Transform torch.Tensor (C, H, W) to PIL image"""
to_PIL = transforms.ToPILImage()

def read_image(path, rgb=False, scale=1.0):
    """Read image from path using PIL.Image

        Returns:
            image (torch.Tensor): image Tensor in (CHW) format.
    """
    image = Image.open(path)
    if rgb:
        image = image.convert('rgb')
    else:
        image = image.convert('L')
    if scale != 1.0:
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.ANTIALIAS)
    return to_tensor(image)

def normalize(input):
    """Normalize pixel values from [0, 255] to [0., 1.]"""
    return input / 255.

def clamp(image, min=0., max=1.):
    """Clamp values in input tensor exceeding (min, max) to (min, max)"""
    return torch.clamp(image, min, max)

def calc_psnr(test_image, target_image, max=1.):
    """Calculate PSNR of a single image.

    Args:
        test_image (torch.Tensor): image to calculate PSNR.
        target_image (torch.Tensor): groud truth batch
        dynamic_range (float): range of pixel values e.g. 1. from [0., 1.].

    Returns:
        psnr (float): average PSNR value.
    """
    mse = mse_loss(test_image, target_image)
    return 20 * torch.log10(max / torch.sqrt(mse)).item()

def generate_noise(size, std, ret_array=False, complex=False):
    """Generate Gaussian noise with mean = 0.

    Note:
        The specified std is used directly here without further normalization.

    Args:
        size (list): shape of the desired noise.
        std (float): standard deviation of noise.

    Returns:
        (torch.Tensor): generated noise.
    """
    if complex:
        """
        noise_real = torch.normal(mean=torch.zeros(*size), std=std)
        noise_imag = torch.normal(mean=torch.zeros(*size), std=std)
        noise = noise_real + 1j * noise_imag
        """
        # Real part and imaginary part are independent. Each has variance = (std ** 2) / 2.
        noise = torch.normal(mean=torch.zeros(*size, dtype=torch.complex64), std=std)
    else:
        noise = torch.normal(mean=torch.zeros(*size), std=std)
    if ret_array:
        return noise.numpy()
    else:
        return noise

def add_noise(image, std, return_noise=False):
    """Add Gaussian noise to image"""
    noise = generate_noise(image.shape, std)
    if return_noise:
        return image + noise, noise
    else:
        return image + noise

def generate_loadlist(datadir, prefix=None, suffix=None, width=None, start_num=1, num_files=None):
    """Generate list of paths to images.

    Note:
        If prefix or suffix are not specified, read everything in a directory, and
        put them in the list in the lexicographic order.
        If prefix and suffix are specified, read files with format [prefix, number, suffix].
        For example, 'test_001.png'.

    Args:
        datadir (str): path to the directory containing images.
        prefix (str): prefix string e.g. 'test_'
        suffix (str): suffix string e.g. '.png'
        width (int): number of digits in the number part of the file names.
            If None, assume that the numbers have no leading 0's.
        start_num (int): the starting number in the names of the images.
        num_files (int): the number of images to read. If none, read all images
            in the directory.

    Returns:
        loadlist: list of paths.
    """
    if num_files is None:
        num_files = len(os.listdir(datadir))
    loadlist = [None] * num_files

    if prefix is None or suffix is None:
        namelist = generate_namelist(datadir, num_files=num_files)
        for i in range(len(namelist)):
            loadlist[i] = os.path.join(datadir, namelist[i])
        return loadlist

    if width is None:
        for i in range(num_files):
            loadlist[i] = os.path.join(datadir, '{prefix}{num:d}{suffix}'.format(prefix = prefix,
                                                                                num = i + start_num,
                                                                                suffix = suffix))
    else:
        for i in range(num_files):
            loadlist[i] = os.path.join(datadir, '{prefix}{num:0{width}}{suffix}'.format(prefix = prefix,
                                                                                  num = i + start_num,
                                                                                  width=width,
                                                                                  suffix = suffix))
    return loadlist

def generate_namelist(datadir, num_files=None, no_exten=False, no_hidden=True):
    """Generate list of file names in a directory

    Note:
        For a specified num_files, the strings are generated in the
            lexicographic order.

    Args:
        datadir (str): path to directory
        num_files (int): number of files to generate the list.
            If None, generate strings of all files in the directory.
    """
    if num_files is None:
        num_files = len(os.listdir(datadir))

    if no_exten:
        namelist = [None] * num_files
        namelist_with_exten = sorted(os.listdir(datadir))[:num_files]
        for i, name in enumerate(namelist_with_exten):
            namelist[i] = os.path.splitext(name)[0]
    else:
        namelist = sorted(os.listdir(datadir))[:num_files]

    if no_hidden:
        namelist = [name for name in namelist if not name.startswith('.')]

    return namelist

def prepare_image_path(datadir, oneim=False):
    """Generate loadlist and namelist"""
    loadlist = generate_loadlist(datadir)
    namelist = generate_namelist(datadir, no_exten=True)
    if oneim:
        loadlist = [loadlist[0]]
        namelist = [namelist[0]]
    num_images = len(loadlist)
    return loadlist, namelist, num_images

def remove_if_exists(path):
    """Remove file if exists

    Args:
        path (str): path to the file.
    """
    try:
        os.remove(path)
    except OSError:
        pass

def mkdir_if_not_exists(path):
    """Make a directory if not already exists

    Args:
        path (str): path to the file.
    """
    if not os.path.exists(path):
        os.mkdir(path)

def load_checkpoint(cpdir, model, optimizer, device=torch.device('cpu')):
    """Load model and optimizer parameters from checkpoint

    Note:
        If optimizer is None, do not load optimizer.
        The checkpoint is expected to be a dict containing theh following keys,
            'model_state_dict': state dict of the model,
            'optimizer_state_dict': state dict of the optimizer,
            'epoch': the epoch count.
            'global_step': the global step count.

    Args:
        cpdir (str): path to the checkpoint.
        model: the model to load the parameters to.
        optimizer: the optimizer to load parameters to.
            If None (e.g. test, deploy, etc.), do not load optimizer.

    Returns:
        start_global_step (int): the global step from the checkpoint.
        start_epoch (int): the epoch from the checkpoint.
    """
    checkpoint = torch.load(cpdir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_global_step = checkpoint['global_step']
    return start_global_step, start_epoch
