# Semantic Segmentation for Remote Sensing

This repo is the PyTorch implementation of some works related to remote sensing tasks.

## Methods

> **SAM-based**:

[SAM_RS](https://arxiv.org/abs/2312.02464) (IEEE TGRS Undergoing Review).


> **Unsupervised Domain Adaptation (UDA)**:

GLGAN (will be released...).

[MBATA_GAN](https://ieeexplore.ieee.org/abstract/document/10032584/) (IEEE TGRS 2023).


> **Multimodal Fusion**:

[FTransUNet](https://ieeexplore.ieee.org/document/10458980) (IEEE TGRS 2024).

[CMFNet](https://ieeexplore.ieee.org/abstract/document/9749821/) (IEEE JSTARS 2022).

[MSFNet](https://ieeexplore.ieee.org/abstract/document/9883789) (IEEE IGARSS 2022).

## Reference
For UDA:
* [TransGAN](https://github.com/VITA-Group/TransGAN)
* [Advent](https://github.com/valeoai/ADVENT)
* [AdaSegNet](https://github.com/wasidennis/AdaptSegNet)

For Semantic Segmentation:
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [UNetformer](https://github.com/WangLibo1995/GeoSeg)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [v-FuseNet](https://github.com/nshaud/DeepNetsForEO)
* [FuseNet](https://github.com/MehmetAygun/fusenet-pytorch)

## Other Works
There are some other works in our group:
> **Change Detection**: [GCD-DDPM](https://github.com/udrs/GCD)

## Citations
If these codes are helpful for your study, please cite:
```
@ARTICLE{ma2024ftransunet,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Liu, Ming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3373033}
}

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
