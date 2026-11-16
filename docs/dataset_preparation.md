# Dataset preparation

We follow [Latte](https://github.com/Vchitect/Latte/blob/main/docs/datasets_evaluation.md) to prepare UCF-101, FaceForensics, SkyTimelapse and Taichi-HD. And we follow [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer/blob/main/datasets/video/realestate10k.py) to prepare RealEstate10K.

## UCF-101

Download [UCF-101 rar file](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and extract it to `data/ucf101`. The dataset structure is like

```
data
└── ucf101
    ├── ApplyEyeMakeup
    │   ├── v_ApplyEyeMakeup_g01_c01.avi
    │   ├── v_ApplyEyeMakeup_g01_c02.avi
    │   └── ...
    ├── ApplyLipstick
    │   ├── v_ApplyLipstick_g01_c01.avi
    │   ├── v_ApplyLipstick_g01_c02.avi
    │   └── ...
    └── ...
```

## FaceForensics, SkyTimelapse and Taichi-HD

Latte provides [FaceForensics](https://huggingface.co/datasets/maxin-cn/FaceForensics), [SkyTimelapse](https://huggingface.co/datasets/maxin-cn/SkyTimelapse/tree/main) and [Taichi-HD](https://huggingface.co/datasets/maxin-cn/Taichi-HD). We further process and repackage the 3 datasets into compact files, released on [Hugging Face](https://huggingface.co/datasets/ZhouTimeMachine/JVT) and [ModelScope](https://modelscope.cn/datasets/ZhouTimeMachine/JVT). You can download and extract them with the following commands:

```bash
hf download --repo-type dataset ZhouTimeMachine/JVT --local-dir data/
# modelscope download --repo-type dataset ZhouTimeMachine/JVT --local_dir data
# ms-hub download --repo-type dataset ZhouTimeMachine/JVT --local-dir data
tar -xzvf data/faceforensics-train.tar.gz -C data/
tar -xzvf data/skytimelapse-train.tar.gz -C data/
tar -xzvf data/taichi-train.tar.gz -C data/
```

## RealEstate10K

The videos of [RealEstate10K](https://google.github.io/realestate10k/) are collected from YouTube, and we do not have the rights to redistribute them as a dataset. To prepare this dataset, please refer to the [DFoT RealEstate10K dataloader](https://github.com/kwsong0113/diffusion-forcing-transformer/blob/main/datasets/video/realestate10k.py).


After the txt annotations from [RealEstate10K website](https://google.github.io/realestate10k/) and the raw videos from Youtube are both downloaded, you can run `scripts/preprocess_re10k.py` to preprocess the dataset. The script will generate the following dataset structure:

```
data
└── re10k
    ├── meta_txts    # txt annotations
    ├── raw_videos   # raw videos from Youtube
    ├── test_256
    │   ├── 0a3b5fb184936a83.mp4
    │   ├── 0a4cf8d9b81b4c6e.mp4
    │   └── ...
    ├── test_poses
    │   ├── 0a3b5fb184936a83.pt
    │   ├── 0a3f289636ba10a7.pt
    │   └── ...
    ├── train_256
    │   ├── 0a0a998c176713fd.mp4
    │   ├── 0a0b1ce8a70f8c0c.mp4
    │   └── ...
    ├── train_poses
        ├── 0a0a998c176713fd.pt
        ├── 0a0b1ce8a70f8c0c.pt
        └── ...
```