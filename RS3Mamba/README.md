# [IEEE GRSL 2024] RS^3Mamba

This repo is the official implementation of ['RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation'](https://ieeexplore.ieee.org/abstract/document/10556777).

![framework](https://github.com/sstary/SSRS/blob/main/docs/RS3Mamba.png)

## Usage
We successfully installed causal_conv1d, mamba_ssm packages according to [VM-UNet](https://github.com/JCruan519/VM-UNet).

We use the ImageNet pretrained VMamba-Tiny model 'vssmtiny_dp01_ckpt_epoch_292.pth' from [Sigma](https://github.com/zifuwan/Sigma), and change the name to 'vmamba_tiny_e292.pth'. The weights is in ./pretrain.

Train the model by: python train_Mamba.py

Please cite our paper if you find it is useful for your research.

```
@ARTICLE{10556777,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={RS3Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}}
  ```
