# L2G



**From Local Matches to Global Masks: Novel Instance Detection in Open-World Scenes**

Qifan Zhang, Sai Haneesh Allu, Jikai Wang, Yangxiao Lu, Yu Xiang

[arXiv](), [Project]()

> Detecting and segmenting novel object instances in open-world environments is a fundamental problem in robotic perception. Given only a small set of template images, a robot must locate and segment a specific object instance in a cluttered, previously unseen scene. Existing proposal-based approaches are highly sensitive to proposal quality and often fail under occlusion and background clutter. We propose L2G-Det, a local-to-global instance detection framework that bypasses explicit object proposals by leveraging dense patch-level matching between templates and the query image. Locally matched patches generate candidate points, which are refined through a candidate selection module to suppress false positives. The filtered points are then used to prompt an augmented Segment Anything Model (SAM) with instance-specific object tokens, enabling reliable reconstruction of complete instance masks. Experiments demonstrate improved performance over proposal-based methods in challenging open-world settings.

## Framework
![L2G.](assets/Framework.png)

## 📸 Detection Examples

### RoboTools

<p align="center">
  <img src="assets/RoboTools.png" width="100%">
</p>

### High Resolution

<p align="center">
  <img src="assets/High_Res.png" width="100%">
</p>


## Getting Started
We prepare demo google colabs: [inference on a high-resolution image](https://colab.research.google.com/drive/1dtlucQ5QryLgooSDkH-Qumxrrnb-9FCg?usp=sharing) and [Training free one-shot detection](https://colab.research.google.com/drive/1IM8TgpNo_9TijopO3PRyZea7MgTUjv30?usp=sharing). 
### Prerequisites
- Python 3.10
- torch (tested 2.6)
- torchvision

### Installation
We test the code on Ubuntu 20.04.
```sh

```

### Preparing Datasets
<details>
<summary> Setting Up 4 Detection Datasets </summary>


#### Dataset

Please put them into "Data" folder as follows:
```
data/
│
├── Query/
│   ├── High_Resolution/
│   │   ├── 000001/
│   │   ├── 000002/
│   │   └── ...
│   │
│   └── RoboTools/
│       ├── 000001/
│       ├── 000002/
│       └── ...
│
└── Templates/
    ├── High_Resolution/
    │   ├── rgb/
    │   │   ├── 000001/
    │   │   ├── 000002/
    │   │   └── ...
    │   └── mask/
    │       ├── 000001/
    │       ├── 000002/
    │       └── ...
    │
    └── RoboTools/
        ├── rgb/
        │   ├── 000001/
        │   ├── 000002/
        │   └── ...
        │
        └── mask/
            ├── 000001/
            ├── 000002/
            └── ...
```


## Acknowledgments

This project is based on the following repositories:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)
- [VoxDet](https://github.com/Jaraxxus-Me/VoxDet)
- [InsDet](https://github.com/insdet/instance-detection)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [SAM6D](https://github.com/JiehongLin/SAM-6D)
- [FFA](https://github.com/s-tian/CUTE)
- [CNOS](https://github.com/nv-nguyen/cnos)