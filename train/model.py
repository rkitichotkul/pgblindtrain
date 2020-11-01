"""Model of CNN-based denoisers.

DnCNN (Zhang et al. 2017, Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising)
-----
The architecture is:

    conv - relu - [conv - batchnorm - relu]x(n-2) - conv

The default number of layer is 20, which is for the blind-denoising model.
The DnCNN model based on https://github.com/SaoYan/DnCNN-PyTorch.

Remez (Remez et al. 2018, Class-Aware Fully-Convolutional Gaussian and Poisson Denoising)
-----
The architecture has two branches. See Fig. 2 in the original paper.

"""

from torch import nn
from torch.nn import functional as F

class DnCNN(nn.Module):
    def __init__(self, num_layers=17):
        super(DnCNN, self).__init__()

        # Fixed parameters
        channels = 1
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
        noise = self.layers(x)
        return x - noise

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class Remez(nn.Module):
    def __init__(self, num_layers=20):
        super(Remez, self).__init__()

        # Fixed parameters
        channels = 1
        kernel_size = 3
        padding = 1         # To maintain the size of image each layer
        features = 63       # Number of features for each convolution in the main branch
        features_n = 1      # Number of features for each "effective noise" convolution

        self.num_layers = num_layers
        main_branch = [None] * num_layers
        noise_branch = [None] * (num_layers + 1)

        main_branch[0] = nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False)
        noise_branch[0] = nn.Conv2d(in_channels=channels, out_channels=features_n,
                                kernel_size=kernel_size, padding=padding, bias=False)
        for i in range(1, num_layers):
            main_branch[i] = nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False)
            noise_branch[i] = nn.Conv2d(in_channels=features, out_channels=features_n,
                                    kernel_size=kernel_size, padding=padding, bias=False)
        noise_branch[num_layers] = nn.Conv2d(in_channels=features, out_channels=features_n,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.main_branch = nn.ModuleList(main_branch)
        self.noise_branch = nn.ModuleList(noise_branch)
        self._initialize_weights()

    def forward(self, x):
        out = self.noise_branch[0](x)
        main = F.relu(self.main_branch[0](x))
        for i in range(1, self.num_layers):
            out += self.noise_branch[i](main)
            main = F.relu(self.main_branch[i](main))
        out += self.noise_branch[self.num_layers](main)
        return out

    def _initialize_weights(self):
        for i in range(self.num_layers):
            self.main_branch[i]._apply(nn.init.kaiming_normal_)
            self.noise_branch[i]._apply(nn.init.kaiming_normal_)
        self.noise_branch[self.num_layers]._apply(nn.init.kaiming_normal_)
