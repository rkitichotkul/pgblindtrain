# README

## Requirements

```
# conda update conda # need this??
# nvcc -V # to check GPU driver version. If 10.1, use the last line. Otherwise, check out pytorch website
conda create -n pgblind python=3.7.8
conda activate pgblind
pip install numpy matplotlib Pillow bm3d h5py tensorboard
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
