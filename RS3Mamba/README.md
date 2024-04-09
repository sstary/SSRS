# [arxiv] RS^3Mamba

This repo is the official implementation of ['RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation'](https://arxiv.org/abs/2404.02457).

![framework](https://github.com/sstary/SSRS/blob/main/docs/RS3Mamba.png)

## Usage
We successfully installed causal_conv1d, mamba_ssm package according to [VM-UNet](https://github.com/JCruan519/VM-UNet).

We use the ImageNet pretrained VMamba-Tiny model from [VMamba](https://github.com/MzeroMiko/VMamba).

Train the model by: python train_Mamba.py

Please cite our paper if you find it is useful for your research.

```
@article{ma2024rs3mamba,
  title={RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation},
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={arXiv preprint arXiv:2404.02457},
  year={2024}}
  ```
