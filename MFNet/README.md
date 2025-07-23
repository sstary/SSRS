# [IEEE TGRS 2025] MFNet

This repo is the official implementation of ['A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation'](https://ieeexplore.ieee.org/abstract/document/11063320).

![framework](https://github.com/sstary/SSRS/blob/main/docs/MFNet.png)

## Usage
You can get the pre-trained model here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

The core modules are in ./MedSAM/models/ImageEncoder and ./MedSAM/models/sam

The current mode is MMLoRA. You can choose MMAdapter or MMLoRA by the 'mod' hyper-parameter in **./MedSAM/cfg.py**, and also need modify the Line 523/524 in **SSRS/MFNet
/UNetFormer_MMSAM.py** and Line 35/36, 42/43 in **SSRS/MFNet/train.py**. Most of the remaining files come from the original SAM framework and can be ignored.


Run the code by: python train.py

Draw the heatmap by: python test_heatmap.py

Please cite our paper if you find it is useful for your research.

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
  ```
