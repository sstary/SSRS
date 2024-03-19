# [arxiv] SAM_RS

This repo is the official implementation of ['SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints'](https://arxiv.org/abs/2312.02464).

![framework](https://github.com/sstary/SSRS/blob/main/docs/SAM_RS.png)

## Usage
We provide image_split.py to split the large patch in ISPRS datasets and the output will be used for SAM pre-processing. The SAM pre-processing results are merged by image_merge.py to get the patch of the original size in ISPRS datasets.

Train the model by: python train.py
Draw the loss by: python draw_loss.py

Please cite our paper if you find it is useful for your research.

```
@article{ma2023sam,
  title={SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints},
  author={Ma, Xianping and Wu, Qianqian and Zhao, Xingyu and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={arXiv preprint arXiv:2312.02464},
  year={2023}}
  ```
