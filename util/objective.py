import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import math
from util import general as gutil
from util.train import dtype

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
def pure(noisy_image, denoised_image, denoiser, alpha, sigma, add_bias=False, device=torch.device('cpu')):
    # reshape to flat vector
    noisy_image_flat = noisy_image.view(-1)
    denoised_image_flat = denoised_image.view(-1)
    n = noisy_image_flat.size()[0]

    # fidelity term
    fidelity = torch.dot(denoised_image_flat, denoised_image_flat)
    fidelity -= 2 * torch.dot(noisy_image_flat, denoised_image_flat)

    # first derivative term
    eps_1 = 1e-4
    random_image_1 = binary_dist(noisy_image.shape, 0.5, [-1, 1])
    random_image_1 = random_image_1.to(device=device, dtype=dtype)
    image_perturb_1 = noisy_image + eps_1 * random_image_1
    denoised_perturb_1 = denoiser(image_perturb_1)
    denoised_perturb_1 = denoised_perturb_1.to(device=device, dtype=dtype)
    first_derivative = torch.dot(random_image_1.view(-1) * (alpha * noisy_image_flat + sigma**2*torch.ones(n)), denoised_perturb_1.view(-1) - denoised_image_flat)
    first_derivative *= 2. / (eps_1)

    # bias term
    bias = 0
    if add_bias:
        bias = torch.dot(noisy_image_flat - alpha * torch.ones(n), noisy_image_flat)
    return (fidelity + first_derivative + bias) / (2 * noisy_image.shape[0])

# Stein-Poisson Unbiased Risk Estimator (SPURE) (assuming H = I)
# noisy image is input vector y
# denoised_image is output vector f(y)
# denoiser is function f
# alpha is scaling factor for Poisson noise
# sigma is standard deviation for additive zero-mean Gaussian noise
# H is point spread function convolution matrix for blurry image
# bias is the constant term, independent of denoiser (can omit when minimizing loss, set add_bias = False)
def spure(noisy_image, denoised_image, denoiser, alpha, sigma, add_bias=False, device=torch.device('cpu')):
    # reshape to flat vector
    noisy_image_flat = noisy_image.view(-1)
    denoised_image_flat = denoised_image.view(-1)
    n = noisy_image_flat.size()[0]

    # fidelity term
    fidelity = torch.dot(denoised_image_flat, denoised_image_flat)
    fidelity -= 2 * torch.dot(noisy_image_flat, denoised_image_flat)

    # first derivative term
    eps_1 = 1e-4
    random_image_1 = binary_dist(noisy_image.shape, 0.5, [-1, 1])
    random_image_1 = random_image_1.to(device=device, dtype=dtype)
    image_perturb_1 = noisy_image + eps_1 * random_image_1
    denoised_perturb_1 = denoiser(image_perturb_1)
    denoised_perturb_1 = denoised_perturb_1.to(device=device, dtype=dtype)
    first_derivative = torch.dot(random_image_1.view(-1) * (alpha * noisy_image_flat + sigma**2*torch.ones(n)), denoised_perturb_1.view(-1) - denoised_image_flat)
    first_derivative *= 2. / (eps_1)

    # second derivative term
    eps_2 = 1e-2
    kappa = 1
    p = 0.5 + (0.5 * kappa / math.sqrt(kappa**2 + 4))
    q = 1 - p
    random_image_2 = binary_dist(noisy_image.shape, p, [math.sqrt(p/q), -math.sqrt(q/p)])
    random_image_2 = random_image_2.to(device=device, dtype=dtype)
    denoised_perturb_2_pos = denoiser(noisy_image + eps_2 * random_image_2)
    denoised_perturb_2_pos = denoised_perturb_2_pos.to(device=device, dtype=dtype)
    denoised_perturb_2_neg = denoiser(noisy_image - eps_2 * random_image_2)
    denoised_perturb_2_neg = denoised_perturb_2_neg.to(device=device, dtype=dtype)
    second_derivative = torch.dot(random_image_2.view(-1), denoised_perturb_2_pos.view(-1) - 2 * denoised_image_flat + denoised_perturb_2_neg.view(-1))
    second_derivative *= -2. * torch.squeeze(alpha * sigma**2) / (kappa * eps_2**2)

    # bias term
    bias = 0
    if add_bias:
        bias = torch.dot(noisy_image_flat - alpha * torch.ones(n), noisy_image_flat) - (sigma**2*n)
    return (fidelity + first_derivative + bias) / (2 * noisy_image.shape[0])

# wrapper function
# usage: example for mse
# in train/main.py: edit from loss_func argument to objective_params
# solve.train(..., objective_params = {"loss": "mse", "alpha": alpha, "sigma": sigma})

# usage: example for pure
# in train/main.py: edit from loss_func argument to objective_params
# solve.train(..., objective_params = {"loss": "pure", "alpha": alpha, "sigma": sigma})
# potentially, add argument --loss to parser to take type of loss as argument

# in train/solve.py: (both cases)
# loss = loss_func(objective_params, image, noisy_image, denoised_image, denoiser)

def loss_func(objective_params, image, noisy_image, denoised_image, denoiser, device=torch.device('cpu')):
    """Wrapper function for loss functions (mse, pure, spure)

    Args:
        objective_params: parameters for calculating loss/objective function.
        image: true image (x)
        noisy_image: image with noise (y)
        denoised_image: image reconstructed by model (f(y))
        denoiser: model used to reconstruct image (f)

    Returns:
        loss
    """

    obj_name = ''
    if type(objective_params) == str:
        if obj_name != 'mse':
            print('Warning: set objective to mse loss')
        obj_name = 'mse'
    elif ('loss' not in objective_params) or (objective_params['loss'] not in ['mse', 'pure', 'spure']):
        obj_name = 'mse'
        print("Warning: set objective to mse loss")
    else:
        obj_name = objective_params['loss']
    if obj_name == 'mse':
        return torch.nn.functional.mse_loss(denoised_image, image, reduction='sum') / (2 * image.shape[0])
    else:
        assert 'alpha' in objective_params, 'Missing parameter: poisson strenth (alpha)'
        assert 'sigma' in objective_params, 'Missing parameter: gaussian std (sigma)'

        add_bias = False
        if 'add_bias' in objective_params:
            add_bias = objective_params['add_bias']

        if obj_name == 'pure':
            return pure(noisy_image, denoised_image, denoiser, objective_params['alpha'], objective_params['sigma'], add_bias, device)
        if obj_name == 'spure':
            return spure(noisy_image, denoised_image, denoiser, objective_params['alpha'], objective_params['sigma'], add_bias, device)
