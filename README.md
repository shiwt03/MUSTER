# UperFormer

UperFormer: Full Transformer-based Networks for Semantic Segmentation

UperFormer structure:
<div align=center><img src="resources/ssformer.jpg"></div>

We use [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.0) as the codebase.

## Installation

For install , please refer to the guidelines in [MMSegmentation v0.29.0](https://github.com/open-mmlab/mmsegmentation/blob/v0.29.0/docs/en/get_started.md#installation).

An example (works for me): ```CUDA 11.6``` and  ```pytorch 1.12.1``` 
### A from-scratch setup script

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMSegmentation.

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
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
Part of UperFormer's segmentation results on Cityscapes:
![image](resources/SSformer_Cityscapes.png)

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
You may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

Part of UperFormer's segmentation results on ADE20K:
![image](resources/SSformer_ADE20K.png)

## Evaluation

Download [trained weights]().

### ADE20K

Example: evaluate ```UperFormer``` on ```ADE20K```:
```shell
# Single-gpu testing
python tools/test.py /path/to/config_file /path/to/checkpoint_file --show
```

### Cityscapes

Example: evaluate ```UperFormer``` on ```Cityscapes```:
```shell
# Single-gpu testing
python tools/test.py /path/to/config_file /path/to/checkpoint_file --show
```

## Training
Download [weights](https://drive.google.com/drive/folders/1oZ4QO0sHhIymh4_8AHz29SXvyofkeIRh?usp=sharing) pretrained on ImageNet-22K, and put them in a folder ```pretrained/```.

Example: train ```UperFormer``` on ```ADE20K```:
```shell
# Single-gpu training
python tools/train.py /path/to/config_file
```
## Visualize
Here is a demo script to test a single image. More details refer to [MMSegmentation's Doc](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).
```
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Example: visualize ```UperFormer``` on ```CityScapes```: 

```shell
coming soon...
```

## License
Please check the LICENSE file. 

## Acknowledgment

Thanks to previous open-sourced repo: [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)