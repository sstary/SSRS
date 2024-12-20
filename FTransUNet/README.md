# [TGRS 2024] FTransUNet

This repo is the official implementation of ['A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation'](https://ieeexplore.ieee.org/document/10458980).

![framework](https://github.com/sstary/SSRS/blob/main/docs/FTransUNet.png)

## Usage
you can get the pre-trained model here: https://console.cloud.google.com/storage/browser/vit_models
in the folder: vit_models/imagenet21k/

Run the code by: python train.py (Train and Test both in this file, but you need to choose the corresponding code.)

If you want to draw the heatmaps by: python test_heatmap.py

Please cite our paper if you find it is useful for your research.

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
  ```
