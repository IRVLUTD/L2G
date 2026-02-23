# L2G



**From Local Matches to Global Masks: Novel Instance Detection in Open-World Scenes**

Qifan Zhang, Sai Haneesh Allu, Jikai Wang, Yangxiao Lu, Yu Xiang

[arXiv](), [Project]()

> Detecting and segmenting novel object instances in open-world environments is a fundamental problem in robotic perception. Given only a small set of template images, a robot must locate and segment a specific object instance in a cluttered, previously unseen scene. Existing proposal-based approaches are highly sensitive to proposal quality and often fail under occlusion and background clutter. We propose L2G-Det, a local-to-global instance detection framework that bypasses explicit object proposals by leveraging dense patch-level matching between templates and the query image. Locally matched patches generate candidate points, which are refined through a candidate selection module to suppress false positives. The filtered points are then used to prompt an augmented Segment Anything Model (SAM) with instance-specific object tokens, enabling reliable reconstruction of complete instance masks. Experiments demonstrate improved performance over proposal-based methods in challenging open-world settings.

## Framework
![L2G.](assets/Framework.png)

## Detection Example
<h1 align="center">📸 Demo Detection Results on RoboTools</h1>

![Demo detection results on RoboTools.](assets/RoboTools.png)


<h1 align="center">📸 Demo detection results on High_Resolution</h1>
![Demo detection results on High_Resolution.](assets/High_Res.png)


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
We do not need training datasets for detectors. We can use template embeddings to train the adapter.

#### High-resolution Dataset
This [instance-detection repo](https://github.com/insdet/instance-detection) provide the [InsDet-Full](https://drive.google.com/drive/folders/1rIRTtqKJGCTifcqJFSVvFshRb-sB0OzP).
```shell
cd $ROOT
ln -s $HighResolution_DATA database
```
We provide the preprocessed testing images in this [link](https://utdallas.box.com/s/bfcgn0dpbvu5w5be20wyj41fzsjajh4h) accorinding to this [instance-detection](https://github.com/insdet/instance-detection). Please put them into "Data" folder as follows:
```
database
│
└───Background
│
└───Objects
│   │
│   └───000_aveda_shampoo
│   │   │   images
│   │   │   masks
│   │
│   └───001_binder_clips_median
│       │   images
│       │   masks
│       │   ...
│   
│   
└───Data
    │   test_1_all
    │   test_1_easy
    │   test_1_hard
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