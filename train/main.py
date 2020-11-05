"""Main program for training/testing CNN denoisers

Perform one of the following actions based on the command line arguments.
    - preprocess
    - train
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from preprocess import generate_datasets
from dataset import DenoiseDataset
from model import DnCNN, Remez
import solve
from util import train as tutil
from util.general import mkdir_if_not_exists

# Setup
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device: {}'.format(device))

# Argument parsing
parser = argparse.ArgumentParser(description='Preprocess or Train/Test CNN denoiser')

# Common arguments
parser.add_argument('mode', type=str, choices=[
                    'preprocess', 'train'], help='Action of this program.')
parser.add_argument('--datadir', type=str,
                    help='Path to images or .h5 files.')

# Arguments for preprocess
parser.add_argument('--numtrain', type=int, default=None,
                    help='Number of train images. If None, use all images in train directory.')
parser.add_argument('--numval', type=int, default=None,
                    help='Number of val images. If None, use all images in val directory.')
parser.add_argument('--numtest', type=int, default=None,
                    help='Number of test images. If None, use all images in test directory.')

# Arguments for train
parser.add_argument('--modeltype', type=str, choices=['dncnn', 'remez'],
                    help='Type of CNN denoiser.')
parser.add_argument('--modeldir', type=str,
                    help='Path to saving checkpoint for training or loading checkpoint for training (including .pth).')
parser.add_argument('--logdir', type=str,
                    help='Path to tensorboard log directory.')
parser.add_argument('--numlayers', type=int, default=17,
                    help='Number of layers in the CNN model.')
parser.add_argument('--logimage', type=int, default=[-1], nargs='*',
                    help='Whether to log images when training/testing.')
parser.add_argument('--sigma', type=int, default=[25], nargs='*',
                    help='STD of Gaussian noise. If 2 values, uniform random STDs between the two values are used.')
parser.add_argument('--alpha', type=int, default=[1000], nargs='*',
                    help='Strength of Poisson noise. If 2 values, uniform random STDs between the two values are used.')
parser.add_argument('--cpdir', type=str, default=None,
                    help='Path to checkpoint to resume model training. If None, train a new model.')
parser.add_argument('--batchsize', type=int, default=100,
                    help='Number of patches in a batch.')
parser.add_argument('--learnrate', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=1, help='Total epochs for training.')
parser.add_argument('--milestone', type=int, default=30, help='Number of epochs after which learning rate is scaled down by 10.')
parser.add_argument('--logevery', type=int, default=10,
                    help='Log loss and PSNR every this number of iterations in an epoch (1 iteration contains #batchsize images).')

args = parser.parse_args()
mode = args.mode
if __name__ == '__main__':
    if mode == 'preprocess':

        print('Parsing training arguments...')
        datadir = args.datadir
        num_train = args.numtrain
        num_val = args.numval
        num_test = args.numtest

        print('Generating datasets...')
        generate_datasets(datadir, num_train=num_train,
                          num_val=num_val, num_test=num_test)

        print('Generating dataset: Done!')

    else: # mode == 'train'

        print('Parsing training arguments...')
        modeltype = args.modeltype
        cpdir = args.cpdir
        datadir = args.datadir
        logdir = args.logdir
        modeldir = args.modeldir
        num_layers = args.numlayers
        sigma = torch.tensor(args.sigma) / 255.
        alpha = torch.tensor(args.alpha)
        batch_size = args.batchsize
        lr = args.learnrate
        epochs = args.epochs
        milestone = args.milestone
        log_every = args.logevery
        log_image = args.logimage

        print('Parameters:')
        print('model type {}, number of layers {}'.format(modeltype, num_layers))
        print('sigma {}, alpha {}'.format(sigma * 255., alpha))
        print('batch size {}, learning rate {}'.format(batch_size, lr))
        print()

        mkdir_if_not_exists(logdir)
        mkdir_if_not_exists(modeldir[:modeldir.rfind('/')])

        print('Setting up training...')
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            tutil.FixedAngleRotation([0, 90, 180, 270]),
            transforms.ToTensor(),
        ])

        dataset_train = DenoiseDataset(os.path.join(datadir, 'train.h5'),
                                       sigma=sigma, alpha=alpha,
                                       transforms=transforms_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        sigma_val, alpha_val = tutil.get_val_params(sigma, alpha)

        dataset_val = DenoiseDataset(os.path.join(datadir, 'val.h5'),
                                     sigma=sigma_val, alpha=alpha_val)
        loader_val = DataLoader(dataset_val)
        dataset_test = DenoiseDataset(os.path.join(datadir, 'test.h5'),
                                      sigma=sigma_val, alpha=alpha_val)
        loader_test = DataLoader(dataset_test)

        if modeltype == 'dncnn':
            model = DnCNN(num_layers=num_layers)
        else: # modeltype = 'remez'
            model = Remez(num_layers=num_layers)
        model = model.to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

        start_global_step, start_epoch = tutil.load_checkpoint_train(cpdir, model, optimizer)

        writer = SummaryWriter(logdir)
        tutil.log_hyperparams(batch_size, sigma, alpha, writer)

        print('Begin training...')
        solve.train(model, loader_train, optimizer, epochs=epochs, scheduler=scheduler,
                    loader_val=loader_val, loader_test=loader_test, device=device, writer=writer,
                    log_every=log_every, log_image=log_image, savedir=modeldir,
                    start_epoch=start_epoch, start_global_step=start_global_step)

        writer.close()
        print('Training: Done!')
