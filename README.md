# TF Keras Stacked Hourglass Networks for Keypoint Estimation

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

An end-to-end single-object keypoint estimation pipeline with Stacked-Hourglass models. Code base is inherited from [Stacked_Hourglass_Network_Keras](https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras) and implement with tf.keras for model training and TFLite for on-device deployment. Support different model type:

- [x] Standard stacked hourglass model
- [x] Mobile stacked hourglass model (using depthwise separable conv)
- [x] Tiny stacked hourglass model (using 192x192 input size and 128 feature channels)
- [x] Configuable stack number


## Guide of train/evaluate/demo

Install requirements on Ubuntu 16.04/18.04:

```
# pip install -r requirements.txt
```

### Train
1. Prepare dataset
    1. MPII Human Pose Dataset
        * Download & extract MPII dataset image package to `data/mpii`:

            ```
            # cd data/mpii && wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
            # tar xzvf mpii_human_pose_v1.tar.gz

            ```
            Images will be placed at `data/mpii/images`
        * Download MPII annotation json file to `data/mpii` and rename to "annotations.json":

    2. MSCOCO Keypoints 2014/2017 Dataset
        * Download & extract MSCOCO train/val dataset image package to `data/mscoco_2014(2017)`:
            ```
            # mkdir -p data/mscoco_2014
            # cd data/mscoco_2014
            # wget http://images.cocodataset.org/zips/train2014.zip
            # wget http://images.cocodataset.org/zips/val2014.zip
            # unzip train2014.zip -d images
            # unzip val2014.zip -d images
            #
            # mkdir -p data/mscoco_2017
            # cd ../data/mscoco_2017
            # wget http://images.cocodataset.org/zips/train2017.zip
            # wget http://images.cocodataset.org/zips/val2017.zip
            # unzip train2017.zip -d images
            # unzip val2017.zip -d images
            ```
        * Download MSCOCO 2014/2017 annotation json file to `data/mscoco_2014(2017)` and rename to "annotations.json"

    3. Customized keypoint dataset
        * Collecting your keypoint images and place to `data/<dataset_name>/images`
        * Generate keypoint annotation json file. The json content is a python list of dicts, for each should include at least following items:

            ```
            {
             'dataset': 'coco',
             'isValidation': 1.0,
             'img_paths': '000000000139.jpg',
             'objpos': [441.4, 217.267],
             'joint_self': [[428.0, 171.0, 1.0], [430.0, 170.0, 2.0], [1.0, 1.0, 0.0], [435.0, 169.0, 2.0], [1.0, 1.0, 0.0], [442.0, 178.0, 2.0], [447.0, 178.0, 2.0], [438.0, 201.0, 2.0], [431.0, 207.0, 2.0], [431.0, 221.0, 2.0], [421.0, 216.0, 2.0], [446.0, 227.0, 2.0], [453.0, 224.0, 2.0], [448.0, 261.0, 2.0], [455.0, 258.0, 2.0], [456.0, 291.0, 2.0], [460.0, 287.0, 2.0]],
             'scale_provided': 0.782
            }
            ```
            Put the annotation file to `data/<dataset_name>` and rename to "annotations.json"
        * Create keypoint config file: classes name file, match point file

            * Classes name file format could refer to  [coco_classes.txt](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/coco_classes.txt). Keypoint order should be aligned with "joint_self" field in annotation json file

            * Match point file format could refer to  [coco_match_point.txt](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/coco_classes.txt). It's used in horizontal/vertical flipping of input image & keypoints for data augment:
                * One row for one pair of matched keypoints in annotation file;
                * Row format: `key_point_name1,key_point_name2,flip_type` (no space). Keypoint name should be aligned with classes name file;
                * Flip type: h-horizontal; v-vertical.

2. [train.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/train.py)

```
# python train.py -h
usage: train.py [-h] [--num_stacks NUM_STACKS] [--mobile] [--tiny]
                [--weights_path WEIGHTS_PATH] [--dataset_path DATASET_PATH]
                [--classes_path CLASSES_PATH]
                [--matchpoint_path MATCHPOINT_PATH] [--batch_size BATCH_SIZE]
                [--optimizer OPTIMIZER] [--learning_rate LEARNING_RATE]
                [--init_epoch INIT_EPOCH] [--total_epoch TOTAL_EPOCH]
                [--gpu_num GPU_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --num_stacks NUM_STACKS
                        number of hourglass stacks, default=2
  --mobile              use depthwise conv in hourglass'
  --tiny                tiny network for speed, input_size=[192x192],
                        channel=128
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --dataset_path DATASET_PATH
                        dataset path containing images and annotation file,
                        default=data/mpii
  --classes_path CLASSES_PATH
                        path to keypoint class definitions,
                        default=configs/mpii_classes.txt
  --matchpoint_path MATCHPOINT_PATH
                        path to matching keypoint definitions for
                        horizontal/vertical flipping image,
                        default=configs/mpii_match_point.txt
  --batch_size BATCH_SIZE
                        batch size for training, default=16
  --optimizer OPTIMIZER
                        optimizer for training (adam/rmsprop/sgd),
                        default=rmsprop
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.0005
  --init_epoch INIT_EPOCH
                        initial training epochs for fine tune training,
                        default=0
  --total_epoch TOTAL_EPOCH
                        total training epochs, default=100
  --gpu_num GPU_NUM     Number of GPU to use, default=1
```

Following is a reference training config cmd:
```
# python train.py --num_stacks=2 --mobile --dataset_path=data/mscoco_2017/ --classes_path=configs/coco_classes.txt --matchpoint_path=configs/coco_match_point.txt
```

Checkpoints during training could be found at `logs/`. Choose a best one as result

You can also use Tensorboard to monitor the loss trend during train:
```
# tensorboard --logdir=logs/
```

MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

### Model dump
We need to dump out inference model from training checkpoint for eval or demo. Following script cmd work for that.

```
# python demo.py --num_stacks=2 --mobile --weights_path=logs/<checkpoint>.h5 --classes_path=configs/coco_classes.txt --dump_model --output_model_file=model.h5
```

Change model type & classes file for different training mode.


### Demo
1. [demo.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/demo.py)
> * Demo script for trained model

(Optional) You can create a skeleton definition file for demo usage. With it you can draw a skeleton on keypoint detection output image. The skeleton file format can refer [coco_skeleton.txt](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/configs/coco_skeleton.txt)

    * One row for one skeleton line;
    * Skeleton line format: `start_keypoint_name,end_keypoint_name,color` (no space). Keypoint name should be aligned with classes name file;
    * Color type: r-red; g-green; b-blue.

image detection mode
```
# python demo.py --num_stacks=2 --mobile --weights_path=model.h5 --classes_path=configs/coco_classes.txt --skeleton_path=configs/coco_skeleton.txt --image
```
video detection mode
```
# python demo.py --num_stacks=2 --mobile --weights_path=model.h5 --classes_path=configs/coco_classes.txt --skeleton_path=configs/coco_skeleton.txt --input=test.mp4
```
For video detection mode, you can use "input=0" to capture live video from web camera and "output=<video name>" to dump out detection result to another video


### Evaluation
Use [eval.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/eval.py) to do evaluation on the inference model with your test dataset. Currently it support PCK (Percentage of Correct Keypoints) metric with fixed normalize coefficient (by default 6.4) on different score threshold. You can also use `--save_result` to save all the detection result on evaluation dataset as images and `--skeleton_path` to draw keypoint skeleton on images:

```
# python eval.py --model_path=model.h5 --dataset_path=data/mscoco_2017/ --classes_path=configs/coco_classes.txt --save_result --skeleton_path=configs/coco_skeleton.txt
```

The default PCK metric (score_threshold=0.5, normalize=6.4) will also be applied on validation dataset during training process for picking best checkpoints.


### Tensorflow model convert
Using [keras_to_tensorflow.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/tools/keras_to_tensorflow.py) to convert the keras .h5 model to tensorflow frozen pb model (only for TF 1.x):
```
# python keras_to_tensorflow.py
    --input_model="path/to/keras/model.h5"
    --output_model="path/to/save/model.pb"
```

You can also use [eval.py](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/eval.py) to do evaluation on the pb inference model


### Inference model deployment
See [on-device inference](https://github.com/david8862/tf-keras-stacked-hourglass-keypoint-detection/blob/master/inference) for TFLite model deployment


### TODO
- [ ] support TF 2.1 fit() API on data generator
- [ ] support more evaluation metrics (PCKh, MSCOCO AP)
- [ ] support more datasets (LSP)
- [ ] support letterbox input image in demo script


## Some issues to know

1. The test environment is
    - Ubuntu 16.04/18.04
    - Python 3.6.8
    - tensorflow 2.0.0/tensorflow 1.14.0
    - tf.keras 2.2.4-tf

2. Training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.


## Contribution guidelines
New features, improvements and any other kind of contributions are warmly welcome via pull request :)


# Citation
Please cite tf-keras-stacked-hourglass-keypoint-detection in your publications if it helps your research:
```
@article{Stacked_Hourglass_Network_Keras,
     Author = {VictorLi},
     Year = {2018}
}
@article{Stacked Hourglass Network,
     title={Stacked Hourglass Networks for Human Pose Estimation},
     author={Alejandro Newell, Kaiyu Yang, Jia Deng},
     journal = {arXiv},
     year={2016}
}
```
