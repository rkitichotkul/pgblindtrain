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
    second_derivative = torch.dot(random_image_2.view(-1), denoised_perturb_2_pos.view(-1) - 2 * denoised_image_flat + denoised_perturb_2_neg.view(-1))
    second_derivative *= -2. * alpha * sigma**2 / (n * kappa * eps_2**2)

    # bias term
    bias = 0
    if add_bias:
        bias = (torch.dot(noisy_image_flat - alpha * torch.ones(n), noisy_image_flat))/n - sigma**2
    return fidelity + first_derivative + bias

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

def loss_func(objective_params, image, noisy_image, denoised_image, denoiser):
    obj_name = ""
    if type(objective_params) == str:
        if obj_name != "mse":
            print("Warning: set objective to mse loss")
        obj_name = "mse"
    elif ("loss" not in objective_params) or (objective_params["loss"] not in ["mse", "pure", "spure"]):
        obj_name = "mse"
        print("Warning: set objective to mse loss")
    else:
        obj_name = objective_params["loss"]
    if obj_name == "mse":
        return torch.nn.MSE_loss(denoised_image, image, reduction='sum') / (2 * noisy_image.shape[0])
    else:
        assert "alpha" in objective_params, "Missing parameter: poisson strenth (alpha)"
        assert "sigma" in objective_params, "Missing parameter: gaussian std (sigma)"

        add_bias = False
        if "add_bias" in objective_params:
            add_bias = objective_params["add_bias"]

        if obj_name == "pure":
            return pure(noisy_image, denoised_image, denoiser, objective_params["alpha"], objective_params["sigma"], add_bias)
        if obj_name == "spure":
            return spure(noisy_image, denoised_image, denoiser, objective_params["alpha"], objective_params["sigma"], add_bias)

# wrapper class for objective function

# usage: example for mse
# initialization obj = Objective("mse")
# forward pass: loss = obj(denoised_image, true_image)
# backward pass: loss.backward()

# usage: example for pure
# initialization obj = Objective("spure", {"alpha":0.01, "std":0.05})
# forward pass: loss = obj(denoised_image, noisy_image, denoiser)
# backward pass: loss.backward()

class Objective:
    def __init__(self, objective = "mse", params = None):
        if objective not in ["mse", "pure", "spure"]:
            print("Invalid choice of objective function, set objective to mse loss")
            objective = "mse"
        if objective == "mse":
            self.obj_name = "mse"
            print("Set objective function: MSE loss")
        else:
            assert params != None, "Missing parameters dictionary"
            assert "alpha" in params, "Missing parameter: poisson strenth (alpha)"
            assert "std" in params, "Missing parameter: gaussian std (std)"
            self.add_bias = False
            if "add_bias" in params:
                self.add_bias = params["add_bias"]
            if objective == "pure":
                self.obj_name = "pure"
                print("Set objective function: PURE")
            if objective == "spure":
                self.obj_name = "spure"
                print("Set objective function: SPURE")

    def calc_loss(self, output, target, denoiser = None):
        if self.obj_name == "mse":
            return torch.nn.MSE_loss(output, target)
        elif self.obj_name == "pure":
            assert "denoiser" is not None, "Missing denoiser"
            return pure(target, output, denoiser, self.alpha, self.std, self.add_bias)
        elif self.obj_name == "spure":
            assert "denoiser" is not None, "Missing denoiser"
            return spure(target, output, denoiser, self.alpha, self.std, self.add_bias)

    def __call__(self, output, target, denoiser = None):
        return self.calc_loss(output, target, denoiser)
