# Training CNN to Denoise Images Corrupted by Mixed Poisson-Gaussian Noise Without Ground Truth Data

This repository contains the code associated with the project "Training CNN to Denoise Images Corrupted by Mixed Poisson-Gaussian Noise Without Ground Truth Data" by Kao Kitichotkul and Patin Inkaew. This project was done as a class project for CS 229: Machine Learning class in the Fall 2020 quarter at Stanford University.

## Requirements

The code has been tested for Python version 3.7.8. The packages required are `numpy`, `matplotlib`, `Pillow`, `bm3d`. `h5py`, `torch` (tested with version 1.7.0), `torchvision` (tested with version 0.8.1), and `tensorboard`.

## Abstract

In this project, we explore the use of unbiased risk estimators as an objective function to train CNN-based denoiser for images corrupted by Poisson and mixed Poisson-Gaussian noise. Our result shows the unbiased risk estimator can be used for the dataset **without ground truth**. The difference between unbiased risk estimator and standard mean square error (MSE) loss during training procedure is only little. Furthermore, the model trained on unbiased risk estimator shows only small decrease in the performance on test set, compared to model trained with MSE with ground truth. Lastly, we demonstrate training model with unbiased risk estimator can be used in transfer learning to improve performance on domain-specific application, such as astrophotography.

## Content

Here are the important files.

* `bash` contains the bash scripts used for preprocessing images and training models by calling `train/main.py`.
* `train` contains the code for preprocessing and training.
  * `main.py` is to be called to preprocess the data or train the model.
  * `preprocess.py`: the code for preprocessing a directory containing images into `.h5` files for training. Note that preprocessing only extract patches of images and pack them into `.h5` files. Noise adding and data augmentation (flipping and rotation) are done in the data loader in `dataset.py`.
  * `model.py`: the definitions of the models used.
  * `solve.py`: the code for training.
  * `dataset.py`: the data loader which reads `.h5` files and feed images (clean and noisy) for training.
* `experiment` contains the code related to testing.
  * `model_demo.py` is for testing models.
* `util` contains utility functions.
  * `denoiser.py` contains wrappers of denoisers for easy usage and some helper functions for setting up CNN denoiser models.
  * `general.py` contains general utility functions.
  * `objective.py` **(IMPORTANT! The point of the project is here.)** contains the implementation of the loss functions i.e. PURE and SPURE which are unbiased estimators of MSE for the Poisson noise case and the Poisson-Gaussian noise case respectively.
  * `train.py` contains utility functions about model training.
* `pgblindtrain.pdf`: the report.

## Citation

```
@misc{pgblindtrain,
      title={Training CNN to Denoise Images Corrupted by Mixed Poisson-Gaussian Noise Without Ground Truth Data}, 
      author={Ruangrawee Kitichotkul and Patin Inkaew},
      year={2020},
      note={CS 229 class project at Stanford University}
}
```



## Contact

Please contact Kao Kitichotkul (rkitichotkul at stanford dot edu) or Patin Inkaew (pinkaew at stanford dot edu) if you have any question.

