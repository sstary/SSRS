# Semantic Segmentation for Remote Sensing

This repo is the PyTorch implementation of some works related to remote sensing tasks.

## Methods

> **SAM-based**:

[MFNet](https://ieeexplore.ieee.org/abstract/document/11063320) (IEEE TGRS 2025)

[SAM_RS](https://ieeexplore.ieee.org/abstract/document/10636322) (IEEE TGRS 2024) (**ESI Highly Cited Paper**).

> **Mamba-based**:

[RS^3Mamba](https://ieeexplore.ieee.org/abstract/document/10556777) (IEEE GRSL 2024) (**ESI Hot Paper**).

> **Unsupervised Domain Adaptation (UDA)**:

[GLGAN](https://ieeexplore.ieee.org/document/10721444) (IEEE TGRS 2024).

[MBATA_GAN](https://ieeexplore.ieee.org/abstract/document/10032584/) (IEEE TGRS 2023) (**ESI Highly Cited Paper**).


> **Multimodal Fusion**:

[MFNet](https://ieeexplore.ieee.org/abstract/document/11063320) (IEEE TGRS 2025)

[ASMFNet](https://ieeexplore.ieee.org/document/10736654) (IEEE JSTARS 2024).

[FTransUNet](https://ieeexplore.ieee.org/document/10458980) (IEEE TGRS 2024) (**ESI Highly Cited Paper**).

[CMFNet](https://ieeexplore.ieee.org/abstract/document/9749821/) (IEEE JSTARS 2022).

[MSFNet](https://ieeexplore.ieee.org/abstract/document/9883789) (IEEE IGARSS 2022).

## Reference
For UDA:
* [TransGAN](https://github.com/VITA-Group/TransGAN)
* [Advent](https://github.com/valeoai/ADVENT)
* [AdaSegNet](https://github.com/wasidennis/AdaptSegNet)

For Segment Anything Model:
* [Medical-SAM-Adapter](https://github.com/MedicineToken/Medical-SAM-Adapter/tree/main)
* [SAM](https://github.com/facebookresearch/segment-anything)

For Semantic Segmentation:
* [TransUNet](https://github.com/Beckschen/TransUNet)
* [UNetformer](https://github.com/WangLibo1995/GeoSeg)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [v-FuseNet](https://github.com/nshaud/DeepNetsForEO)
* [FuseNet](https://github.com/MehmetAygun/fusenet-pytorch)

For Semantic Segmentation based on Mamba:
* [VMamba](https://github.com/MzeroMiko/VMamba)
* [VM-UNet](https://github.com/JCruan519/VM-UNet)
* [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba)

## Datasets
All datasets including ISPRS Potsdam, ISPRS Vaihingen, loveDA can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets).

## Scripts
There are some scripts in utils folder including: split images, merge images, draw loss curves.

## Other Works
There are some other works in our group:
> **Scene Classification**: [DF4LCZ](https://github.com/ctrlovefly/DF4LCZ) (IEEE TGRS 2024).

> **Change Detection**: [GCD-DDPM](https://github.com/udrs/GCD) (IEEE TGRS 2024) (**ESI Highly Cited Paper**), [GVLM](https://github.com/zxk688/GVLM) (ISPRS 2023) (**ESI Highly Cited Paper**).

> **Super-resolution**: [ASDDPM](https://github.com/littlebeen/ASDDPM-Adaptive-Semantic-Enhanced-DDPM) (IEEE JSTARS 2024), [GCRDN](https://github.com/zxk688/GCRDN) (IEEE JSTARS 2023).

## Citations
If these codes are helpful for your study, please cite:
```
@ARTICLE{ma2024manet,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15}
}

@article{ma2024frequency,
  author={Ma, Xianping and Zhang, Xiaokang and Ding, Xingchen and Pun, Man-On and Ma, Siwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Decomposition-Based Unsupervised Domain Adaptation for Remote Sensing Image Semantic Segmentation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-18}
}

@ARTICLE{ma2024rs3mamba,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={RS$^3$Mamba: Visual State Space Model for Remote Sensing Image Semantic Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}
}

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

@article{ma2024sam,
  author={Ma, Xianping and Wu, Qianqian and Zhao, Xingyu and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SAM-Assisted Remote Sensing Imagery Semantic Segmentation With Object and Boundary Constraints}, 
  year={2024},
  volume={62},
  pages={1-16},
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

@ARTICLE{ma2024asmfnet,
  author={Ma, Xianping and Xu, Xichen and Zhang, Xiaokang and Pun, Man-On},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Adjacent-Scale Multimodal Fusion Networks for Semantic Segmentation of Remote Sensing Data}, 
  year={2024},
  volume={},
  number={},
  pages={1-13}
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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sstary/SSRS&type=Date)](https://star-history.com/#sstary/SSRS&Date)
