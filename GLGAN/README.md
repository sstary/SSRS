# [IEEE TGRS 2024] GLGAN/De-GLGAN

This repo is the official implementation of ['Decomposition-based Unsupervised Domain Adaptation for Remote Sensing Image Semantic Segmentation'](https://ieeexplore.ieee.org/document/10721444).

![framework](https://github.com/sstary/SSRS/blob/main/docs/GLGAN.png)

## Usage
Please download the official pre-trained weights from ['Swin Transformer'](https://github.com/microsoft/Swin-Transformer)--'ImageNet-1K and ImageNet-22K Pretrained Swin-V2 Models'--['SwinV2-B*'](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth) and ['SwinV2-L*'](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth). The last cloumn '1K model' is what we used.

Run the GLGAN by: python GLGAN_*.py

Run the decomposition-based GLGAN by: python FDGLGAN_*.py

Please cite our paper if you find it is useful for your research.

```
@article{ma2024frequency,
  author={Ma, Xianping and Zhang, Xiaokang and Ding, Xingchen and Pun, Man-On and Ma, Siwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Decomposition-Based Unsupervised Domain Adaptation for Remote Sensing Image Semantic Segmentation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-18}
}
  ```

