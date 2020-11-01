import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import math
from util import general as gutil

# binary distribution
# p(value[0]) = 1 - prob and p(value[1]) = prob
def binary_dist(sample_shape, prob, value=[0, 1]):
    dist = torch.distributions.Bernoulli(torch.tensor([prob]))
    label = dist.sample(sample_shape).view(sample_shape)
    sample = torch.zeros(sample_shape)
    sample[label == 0] = value[0]
    sample[label == 1] = value[1]
    return sample

# Poisson Unbiased Risk Estimator (PURE) (assuming H = I)
# noisy image is input vector y
# denoised_image is output vector f(y)
# denoiser is function f
# alpha is scaling factor for Poisson noise
# sigma is standard deviation for additive zero-mean Gaussian noise
# H is point spread function convolution matrix for blurry image
# bias is the constant term, independent of denoiser (can omit when minimizing loss, set add_bias = False)
def pure(noisy_image, denoised_image, denoiser, alpha, sigma, add_bias=False):
    # reshape to flat vector
    noisy_image_flat = noisy_image.view(-1)
    denoised_image_flat = denoised_image.view(-1)
    n = noisy_image_flat.size()[0]

    # fidelity term
    fidelity = 1/n * torch.dot(denoised_image_flat, denoised_image_flat)
    fidelity -= 2/n * torch.dot(noisy_image_flat, denoised_image_flat)

    # first derivative term
    eps_1 = 1e-4
    random_image_1 = binary_dist(noisy_image.shape, 0.5, [-1, 1])
    image_perturb_1 = noisy_image + eps_1 * random_image_1
    denoised_perturb_1 = denoiser(image_perturb_1, std=sigma)
    first_derivative = torch.dot(random_image_1.view(-1) * (alpha * noisy_image_flat + sigma**2*torch.ones(n)), denoised_perturb_1.view(-1) - denoised_image_flat)
    first_derivative *= 2. / (n * eps_1)

    # bias term
    bias = 0
    if add_bias:
        bias = (torch.dot(noisy_image_flat - alpha * torch.ones(n), noisy_image_flat))/n
    return fidelity + first_derivative + bias

# Stein-Poisson Unbiased Risk Estimator (SPURE) (assuming H = I)
# noisy image is input vector y
# denoised_image is output vector f(y)
# denoiser is function f
# alpha is scaling factor for Poisson noise
# sigma is standard deviation for additive zero-mean Gaussian noise
# H is point spread function convolution matrix for blurry image
# bias is the constant term, independent of denoiser (can omit when minimizing loss, set add_bias = False)
def spure(noisy_image, denoised_image, denoiser, alpha, sigma, add_bias=False):
    # reshape to flat vector
    noisy_image_flat = noisy_image.view(-1)
    denoised_image_flat = denoised_image.view(-1)
    n = noisy_image_flat.size()[0]

    # fidelity term
    fidelity = 1/n * torch.dot(denoised_image_flat, denoised_image_flat)
    fidelity -= 2/n * torch.dot(noisy_image_flat, denoised_image_flat)

    # first derivative term
    eps_1 = 1e-4
    random_image_1 = binary_dist(noisy_image.shape, 0.5, [-1, 1])
    image_perturb_1 = noisy_image + eps_1 * random_image_1
    denoised_perturb_1 = denoiser(image_perturb_1, std=sigma)
    first_derivative = torch.dot(random_image_1.view(-1) * (alpha * noisy_image_flat + sigma**2*torch.ones(n)), denoised_perturb_1.view(-1) - denoised_image_flat)
    first_derivative *= 2. / (n * eps_1)

    # second derivative term
    eps_2 = 1e-2
    kappa = 1
    p = 0.5 + (0.5 * kappa / math.sqrt(kappa**2 + 4))
    q = 1 - p
    random_image_2 = binary_dist(noisy_image.shape, p, [math.sqrt(p/q), -math.sqrt(q/p)])
    denoised_perturb_2_pos = denoiser(noisy_image + eps_2 * random_image_2, std=sigma)
    denoised_perturb_2_neg = denoiser(noisy_image - eps_2 * random_image_2, std=sigma)
    second_derivative = torch.dot(random_image_2.view(-1) * torch.ones(n), denoised_perturb_2_pos.view(-1) - 2 * denoised_image_flat + denoised_perturb_2_neg.view(-1))
    second_derivative *= -2. * alpha * sigma**2 / (n * kappa * eps_2**2)

    # bias term
    bias = 0
    if add_bias:
        bias = (torch.dot(noisy_image_flat - alpha * torch.ones(n), noisy_image_flat))/n - sigma**2
    return fidelity + first_derivative + bias
