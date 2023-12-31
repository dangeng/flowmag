# Self-Supervised Motion Magnification by Backpropagating Through Optical Flow
[Zhaoying Pan*](https://zhaoyingpan.github.io/), [Daniel Geng*](https://dangeng.github.io/), [Andrew Owens](http://andrewowens.com/), NeurIPS 2023.

(* Equal Contributions)


#### [[arXiv]](https://arxiv.org/abs/2311.17056) | [[Video]](https://www.youtube.com/watch?v=ik_zVqMrJh8) | [[Website]](https://dangeng.github.io/flowmag)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TmD1SGFlxtodDGWzYLjAiDBvRod2u7eZ?usp=sharing)

## Table of contents
- [Introduction](#introduction)
- [Colab notebook](#colab-notebook)
- [Installation](#installation)
- [Usage](#usage)
- [Additional sections](#additional-sections)
    - [Dataset and test videos](#dataset-and-test-videos)
    - [Optical flow checkpoints](#optical-flow-checkpoints)
    - [Train your model](#train-your-model)
    - [Targeted magnification](#targeted-magnification)
    - [Test-time adaptation](#test-time-adaptation)
    - [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction

![Overview](./data/overview.png)

We present a simple, self-supervised method for magnifying subtle motions in video: given an input video and a magnification factor, we manipulate the video such that its new optical flow is scaled by the desired amount. To train our model, we propose a loss function that estimates the optical flow of the generated video and penalizes how far if deviates from the given magnification factor. Thus, training involves differentiating through a pretrained optical flow network. Since our model is self-supervised, we can further improve its performance through test-time adaptation, by finetuning it on the input video. It can also be easily extended to magnify the motions of only user-selected objects. Our approach avoids the need for synthetic magnification datasets that have been used to train prior learning-based approaches. Instead, it leverages the existing capabilities of off-the-shelf motion estimators.

## Colab notebook

Try our method in this [colab notebook](https://colab.research.google.com/drive/1TmD1SGFlxtodDGWzYLjAiDBvRod2u7eZ?usp=sharing). We provide <i>Butterfly</i> and <i>Cats</i> sequences as examples for magnification with/without targeting respectively. This colab notebook supports uploading videos for magnification with/without targeting as well. By plotting bounding box with the widget, a mask can be generated with Segment Anything Model and applied for magnification.

## Installation

### Environment

We use `python 3.7.4` for our experiments.

After cloning our repo, please run
```
cd flowmag
pip install -r requirements.txt
```

### Downloading checkpoint

We provide the checkpoint of model implemented with RAFT or ARFlow, trained for 140 epochs. 
Download the [model (RAFT)](https://drive.google.com/file/d/1ESSaea-Roe1feFugPFycW5Dd7QCg2ZXR/view?usp=sharing) or the [model (ARFlow)](https://drive.google.com/file/d/1m-nE_-3AJ549W3Yemnrm4XeR28tP1sUM/view?usp=sharing) from google drive. Run the following command to download both checkpoints.
```
sh checkpoints/download_models.sh
```

## Usage

### Inference

To run inference, here is a sample command:

```
python inference.py \
    --config configs/alpha16.color10.yaml \
    --frames_dir ./data/example \
    --resume ./checkpoints/raft_chkpt_00140.pth \
    --save_name example.raft.ep140 \
    --alpha 20 \
    --output_video
```

Here are the possible arguments: 

- `--config` is the path to the config of the model
- `--frames_dir` is a path to a directory of frames
- `--resume` is path to the checkpoint
- `--save_name` is the name to save under (will be automatically saved to the log file of the experiment under `[log_dir]/inference/[save_name]`)
- `--alpha` is the magnification factor
- `--output_video` flag saves the magnified video to a mp4 file, otherwise, the magnified frames will be saved as image files.

The magnified video will be saved to `./inference/example/x20.mp4`.


## Additional sections

### Dataset and test videos

We collected a dataset containing 145k unlabeled frame pairs from several public datasets, 
including Youtube-VOS-2019, DAVIS, Vimeo-90k, Tracking Any Object (TAO), and Unidentified 
Video Objects (UVO). The code for data collection will be updated in this repo soon. The 
collected dataset has a size of ~80GB, and you should be able to re-generate the same dataset 
with `sh scripts/prepare_data.sh` (may modify the root directory in it to avoid insufficient space). We provide 
the zip file of [test data](https://drive.google.com/file/d/1e9KljPpIHB5Yq8r2-XcHLEHlym6n1H5C/view?usp=sharing), 
containing a folder of images named `test` and a json file of image filenames named `test_fn.json`. The generated dataset 
should contain the following folder structure:

```
flowmag_data/
│
├── train/
│   ├── frameA/
│   └── frameB/
│
├── test/
│   ├── frameA/
│   └── frameB/
│
├── train_fn.json
│
└── test_fn.json
```

Put the generated dataset in this path `./data/flowmag_data`, or modify the `dataroot` in the config file to your dataset directory.

We provide original videos used in our experiments at the [Google Drive folder](https://drive.google.com/drive/folders/12kidhGIosh_8MJpXnCTiY1w4uJCcqwyG?usp=sharing). 
During inference, our model takes a folder of image files of video frames. To convert the mp4 file
into a folder of images, you may use this command.

```
ffmpeg -i /video_root/video_name.mp4 /image_root/video_name/%04d.png
```

Replace `video_root`, `image_root` with the root folder paths of your videos and image folder.
Replace `video_name` with your video name. 
Example command: `ffmpeg -i ./videos/twocats.mp4 ./images/twocats/%04d.png`.

We provide a short clip of `twocats` as an example in our repo (`./data/example`). The whole video has 261 frames in total. For 
flexibility, we only store the first 20 frames as an example in our repo. If you are interested,
please check our [Google Drive folder](https://drive.google.com/drive/folders/12kidhGIosh_8MJpXnCTiY1w4uJCcqwyG?usp=sharing) for the video file.


### Optical flow checkpoints

For optical flow calculation, we provide two options (RAFT, ARFlow) for training the model, and four
options (PWC-Net, RAFT, GMFlow with two checkpoints) for evaluation. Please download the checkpoints
from the following links and put the checkpoint file in the folders accordingly.

| Flow Model  | Checkpoint Folder |  Downloading Link |
| --- | --- | --- |
| RAFT |   ./flow_models/raft   | https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view?usp=drive_link   |
| ARFlow  |   ./flow_models/ARFlow/checkpoints/KITTI15   |   https://github.com/lliuz/ARFlow/blob/master/checkpoints/KITTI15/pwclite_ar.tar   |
| GMFlow    |   ./flow_models/gmflow   |  https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view?usp=sharing   |

For GMFlow, please unzip the file and use `gmflow_sintel-0c07dcb3.pth` or `gmflow_things-e9887eda.pth` for evaluation.

### Train your model

We provide two config files for training models with RAFT/ARFlow on 4 A40 GPUs. Here is the sample command of training model with RAFT.

```
python train.py --config configs/alpha16.color10.yaml
```
- `--config` is the path to the config of the model

The default training setting uses 4 A40 GPUs with a batchsize of 40, and please adjust the settings accordingly.

The config.yaml, logs.txt, checkpoints and etc will be saved under the folder `.results/timestamp-alpha16.color10.raft`.
You may change the default folder in `log_dir` in the config file.

If you wish to finetune the model, modify the `resume` in config file to the model path on which you want to finetune.

### Targeted magnification

With a given mask saved in npy file, our method is capable to magnify the motions of only user-selected objects.

```
python inference.py \
    --config configs/alpha16.color10.yaml \
    --frames_dir ./data/example \
    --resume ./checkpoints/raft_chkpt_00140.pth \
    --save_name example.raft.ep140 \
    --alpha 20 \
    --mask_path ./data/white_cat_mask.npy \
    --soft_mask 25 \
    --output_video
```

Here are the possible arguments: 

- `--config` is the path to the config of the model
- `--frames_dir` is a path to a directory of frames
- `--resume` is path to the checkpoint
- `--save_name` is the name to save under (will be automatically saved to the log file of the experiment under `[log_dir]/inference/[save_name]`)
- `--alpha` is the magnification factor
- `--mask_path` is the path to the npy file of the mask
- `--soft_mask` is the parameter for softing the mask
- `--output_video` flag saves the magnified video to a mp4 file, otherwise, the magnified frames will be saved as image files.

### Test-time adaptation

It is feasible to finetune our model on the input video to achieve better performance on the video.

```
python inference.py \
    --config configs/alpha16.color10.yaml \
    --frames_dir ./data/example \
    --resume ./checkpoints/raft_chkpt_00140.pth \
    --save_name example.raft.ep140 \
    --alpha 20 \
    --test_time_adapt \
    --tta_epoch 3 \
    --output_video
```

Here are the possible arguments: 

- `--config` is the path to the config of the model
- `--frames_dir` is a path to a directory of frames
- `--resume` is path to the checkpoint
- `--save_name` is the name to save under (will be automatically saved to the log file of the experiment under `[log_dir]/inference/[save_name]`)
- `--alpha` is the magnification factor
- `--test_time_adapt` is the flag to enable test time adaptation
- `--tta_epoch` is the number of epoch for test time adaptation
- `--output_video` flag saves the magnified video to a mp4 file, otherwise, the magnified frames will be saved as image files.

### Evaluation

We provide the code to evaluate the models with the metrics, motion error, and magnification error.

```
python eval.py \
    --config configs/alpha16.color10.yaml \
    --resume ./checkpoints/raft_chkpt_00140.pth \
    --alpha 32 \
    --flow_model gmflow \
    --flow_model_type things
```

Here are the possible arguments: 

- `--config` is the path to the config of the model
- `--resume` is path to the checkpoint
- `--flow_model` is the flow model used for flow calculation (available options: `pwcnet`, `raft`, `gmflow`)
- `--flow_model_type` is checkpoint used for the optical flow model, and available options can be checked in the following table

|   flow_model     |    flow_model_type     |
| --- | --- |
| pwcnet | na |
| raft | things |
| gmflow | sintel |
| gmflow | things |

If you wish to use PWC-Net for evaluation, please see this [link](https://pypi.org/project/cupy/) to install `cupy`.
Remember to check your cuda and cudatoolkits version before you install it. We use `cupy-cuda110==8.3.0`.

The evaluated results will be saved in a txt file named `eval_results/alpha16.color10.ep140/flowmag_gmflow_things.txt`.

## Citation

If you found this code useful please consider citing our [paper](https://arxiv.org/abs/2311.17056):

```
@inproceedings{pan2023selfsupervised,
  title={Self-Supervised Motion Magnification by Backpropagating Through Optical Flow},
  author={Zhaoying Pan and Daniel Geng and Andrew Owens},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=hLPJ7xLbNF}
}
```


## Acknowledgements
* [UNet](https://github.com/milesial/Pytorch-UNet/)
* [RAFT](https://github.com/princeton-vl/RAFT)
* [ARFlow](https://github.com/lliuz/ARFlow)
* [GMFlow](https://github.com/haofeixu/gmflow)
* [CorrWise Loss](https://github.com/dangeng/CorrWiseLosses)
