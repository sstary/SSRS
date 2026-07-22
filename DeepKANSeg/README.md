# [IEEE TGRS 2026] DeepKANSeg

This repo is the official implementation of ['Kolmogorov-Arnold Network for Remote Sensing Image Semantic Segmentation'](https://ieeexplore.ieee.org/abstract/document/11615317).

![framework](https://github.com/sstary/SSRS/blob/main/docs/DeepKANSeg.png)

## Usage
Switch the backbone by changing the comment: 

from model.deepKAN_ResNet import UNetFormer as DeepKANSeg

from model.deepKAN_SAM import UNetFormer as DeepKANSeg

Train the model by: python train.py

Please cite our paper if you find it is useful for your research.

```
@ARTICLE{wang2026KAN,
  author={Wang, Ziyao and Hu, Yin and Zhang, Xiaokang and Ye, Yuanxin and Ma, Xianping and Pun, Man-On},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Kolmogorov-Arnold Network for Remote Sensing Image Semantic Segmentation}, 
  year={2026},
  pages={1-1}
}
  ```
