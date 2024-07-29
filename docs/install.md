# Installation Instruction

In this document, we provide instructions of how to properly install this codebase. We highly recommend using a conda environment to simplify set up.

## Setup Conda Environment

You can skip this section if you are not using conda virtual environment.

Note: `pytorch` will be installed together with IsaacGym (please refer to [pytorch](https://pytorch.org/get-started/locally/)). There are certain compatibility requirement for IsaacGym.

```
conda create -y -n penspin python=3.8
conda activate penspin
conda install -c conda-forge urllib3
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r  gdown hydra-core termcolor gym tensorboardX trimesh numpy==1.22.4 wandb
```

## PenSpin

```
git clone https://github.com/HaozhiQi/penspin
cd penspin
pip install -r requirements.txt
```

## IsaacGym

Download IsaacGym Preview 4.0 ([Download](https://developer.nvidia.com/isaac-gym)), then follow the installation instructions in the documentation. We provide the bash commands what we did during development.

You can also download from our google drive by
```
gdown 1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9 -O ISAAC_PATH
cd ISAAC_PATH
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```
