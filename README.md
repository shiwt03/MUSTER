# UperFormer

UperFormer: A Multi-scale Transformer-based Decoder for Semantic Segmentation

The core codes can be found at:
```
UperFormer/mmseg/models/decode_heads/UperFormer_head.py
```

We use [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.0) as the codebase.

## Installation

For install , please refer to the guidelines in [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/blob/v0.29.0/docs/en/get_started.md#installation).

An example (works for me): ```CUDA 11.6``` and  ```pytorch 1.12.1```

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install UperFormer.

```shell
# The workdir of the terminal should be UperFormer/
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Dataset Preparation

For dataset preparation, please refer to the guidelines in this [link](https://github.com/open-mmlab/mmsegmentation/blob/v0.29.0/docs/en/dataset_prepare.md#prepare-datasets).

It is recommended to symlink the dataset root to `UperFormer/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

The fold structure is recommended to be:
```none
UperFormer
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
MMsegmentation provided a [script](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
You may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

## Evaluation

Because of anonymity, we cannot give trained weights at present.

### ADE20K

Example: evaluate ```UperFormer``` with ```Swin Transformer``` on ```ADE20K```:
```shell
# Single-gpu testing
python tools/test.py configs/UperFormer/_UperFormer_swin-base_ade20k_160k.py /path/to/checkpoint_file --show
```

### Cityscapes

Example: evaluate ```UperFormer``` with ```Swin Transformer``` on ```Cityscapes```:
```shell
# Single-gpu testing
python tools/test.py configs/UperFormer/_UperFormer_swin-base_ade20k_160k.py /path/to/checkpoint_file --show
```

## Training

Example: train ```UperFormer``` with ```Swin Transformer``` on ```ADE20K```:
```shell
# Single-gpu training
python tools/train.py configs/UperFormer/_UperFormer_swin-base_ade20k_160k.py
```

## License
Please check the LICENSE file.

## Citation
Please check ```CITATION.cff```.