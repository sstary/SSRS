# Semantic Segmentation for Remote Sensing

This repo is the pyTorch implementation of some works related to remote sensing tasks. It currently includes code and models for the following tasks:
> **SAM-based**:

SAM_RS: [SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints](https://arxiv.org/abs/2312.02464) (TGRS Undergoing Review).

> **Unsupervised Domain Adaptation (UDA)**:

GLGAN: Decomposition-based Unsupervised Domain Adaptation for Remote Sensing Images Semantic Segmentation (will be released...).

MBATA_GAN: [Unsupervised Domain Adaptation Augmented by Mutually Boosted Attention for Semantic Segmentation of VHR Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/10032584/) (TGRS).

> **Multimodal Fusion**:

FTransUNet: A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation (will be released...) (TGRS Undergoing Review).

CMFNet: [A Crossmodal Multiscale Fusion Network for Semantic Segmentation of Remote Sensing Data](https://ieeexplore.ieee.org/abstract/document/9749821/) (JSTARS).

MSFNet: [MSFNET: MULTI-STAGE FUSION NETWORK FOR SEMANTIC SEGMENTATION OF FINE-RESOLUTION REMOTE SENSING DATA](https://ieeexplore.ieee.org/abstract/document/9883789) (IGARSS 2022).

## Reference

-- For UDA:
* TransGAN: https://github.com/VITA-Group/TransGAN
* Advent: https://github.com/valeoai/ADVENT
* AdaSegNet: https://github.com/wasidennis/AdaptSegNet

-- For Semantic Segmentation:
* UNetformer: https://github.com/WangLibo1995/GeoSeg
* UCTransNet: https://github.com/McGregorWwww/UCTransNet
* Swin-Transformer: https://github.com/microsoft/Swin-Transformer
* Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet
* v-FuseNet: https://github.com/nshaud/DeepNetsForEO
* FuseNet: https://github.com/MehmetAygun/fusenet-pytorch

## Citations
If these codes are helpful for your study, please cite:
```
@article{ma2023sam,
  title={SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints},
  author={Ma, Xianping and Wu, Qianqian and Zhao, Xingyu and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={arXiv preprint arXiv:2312.02464},
  year={2023}
}

@article{ma2023unsupervised,
  title={Unsupervised domain adaptation augmented by mutually boosted attention for semantic segmentation of vhr remote sensing images},
  author={Ma, Xianping and Zhang, Xiaokang and Wang, Zhiguo and Pun, Man-On},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--15},
  year={2023},
  publisher={IEEE}
}

@article{ma2022crossmodal,
  title={A crossmodal multiscale fusion network for semantic segmentation of remote sensing data},
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={15},
  pages={3463--3474},
  year={2022},
  publisher={IEEE}
}

@inproceedings{ma2022msfnet,
  title={MSFNET: Multi-Stage Fusion Network for Semantic Segmentation of Fine-Resolution Remote Sensing Data},
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Liu, Ming},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={2833--2836},
  year={2022},
  organization={IEEE}
}
```

## Contact 
Xianping Ma ([xianpingma@ling.cuhk.edu.cn](xianpingma@ling.cuhk.edu.cn)), ([ma.xianping125@gmail.com](haonan1wang@gmail.com))
