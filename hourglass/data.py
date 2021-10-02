#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random
import numpy as np
from PIL import Image
import json
from tensorflow.keras.utils import Sequence

from common.data_utils import random_horizontal_flip, random_vertical_flip, random_brightness, random_grayscale, random_chroma, random_contrast, random_sharpness, random_blur, random_histeq, random_rotate_angle, crop_single_object, rotate_single_object, crop_image, normalize_image, transform_keypoints, generate_gt_heatmap

# by default, Stacked Hourglass model use output_stride = 4, which means:
#
#   input image:    256 x 256 x 3
#   output heatmap: 64 x 64 x num_classes
#
HG_OUTPUT_STRIDE = 4


class hourglass_dataset(Sequence):
    def __init__(self, dataset_path,
                       batch_size,
                       class_names,
                       input_shape,
                       num_hgstack=2,
                       is_train=True,
                       with_meta=False,
                       matchpoints=None):
        self.json_file = os.path.join(dataset_path, 'annotations.json')
        self.image_path = os.path.join(dataset_path, 'images')
        self.batch_size = batch_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.input_shape = input_shape
        self.num_hgstack = num_hgstack
        self.is_train = is_train
        self.with_meta = with_meta
        # output heatmap size should be 1/HG_OUTPUT_STRIDE of input size
        self.output_shape = (self.input_shape[0]//HG_OUTPUT_STRIDE, self.input_shape[1]//HG_OUTPUT_STRIDE)
        self.dataset_name = None
        self.train_annotations, self.val_annotations = self._load_image_annotation()
        if self.is_train:
            self.annotations = self.train_annotations
        else:
            self.annotations = self.val_annotations

        self.horizontal_matchpoints, self.vertical_matchpoints = self._get_matchpoint_list(matchpoints)

        # Preallocate memory for batch input/output data
        # input images:    batch_size * input_shape  * channel (3)
        # output heatmaps: batch_size * output_shape * num_classes
        self.batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        self.batch_heatmaps = np.zeros(shape=(self.batch_size, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        self.batch_metainfo = list()

    def _get_matchpoint_list(self, matchpoints):
        horizontal_matchpoints, vertical_matchpoints = [], []
        if matchpoints:
            for matchpoint_line in matchpoints:
                #matchpoint line format: [key_point_name1,key_point_name2,flip_type]
                matchpoint_list = matchpoint_line.split(',')
                matchpoint_pair = (self.class_names.index(matchpoint_list[0]), self.class_names.index(matchpoint_list[1]))
                if matchpoint_list[2] == 'h':
                    horizontal_matchpoints.append(matchpoint_pair)
                elif matchpoint_list[2] == 'v':
                    vertical_matchpoints.append(matchpoint_pair)
                else:
                    raise ValueError('invalid flip type')

        return horizontal_matchpoints, vertical_matchpoints


    def _load_image_annotation(self):
        # load train or val annotation
        #
        # MPII annotation is a list of following format dict:
        # {'dataset': 'MPI',
        #  'isValidation': 0.0,
        #  'img_paths': '015601864.jpg',
        #  'img_width': 1280.0,
        #  'img_height': 720.0,
        #  'objpos': [594.0, 257.0],
        #  'joint_self': [[620.0, 394.0, 1.0], [616.0, 269.0, 1.0], [573.0, 185.0, 1.0], [647.0, 188.0, 0.0], [661.0, 221.0, 1.0], [656.0, 231.0, 1.0], [610.0, 187.0, 0.0], [647.0, 176.0, 1.0], [637.02, 189.818, 1.0], [695.98, 108.182, 1.0], [606.0, 217.0, 1.0], [553.0, 161.0, 1.0], [601.0, 167.0, 1.0], [692.0, 185.0, 1.0], [693.0, 240.0, 1.0], [688.0, 313.0, 1.0]],
        #  'scale_provided': 3.021,
        #  'joint_others': [[895.0, 293.0, 1.0], [910.0, 279.0, 1.0], [945.0, 223.0, 0.0], [1012.0, 218.0, 1.0], [961.0, 315.0, 1.0], [960.0, 403.0, 1.0], [979.0, 221.0, 0.0], [906.0, 190.0, 0.0], [912.491, 190.659, 1.0], [830.509, 182.341, 1.0], [871.0, 304.0, 1.0], [883.0, 229.0, 1.0], [888.0, 174.0, 0.0], [924.0, 206.0, 1.0], [1013.0, 203.0, 1.0], [955.0, 263.0, 1.0]],
        #  'scale_provided_other': 2.472,
        #  'objpos_other': [952.0, 222.0],
        #  'annolist_index': 5.0,
        #  'people_index': 1.0,
        #  'numOtherPeople': 1.0,
        #  'headboxes': [[0.0, 0.0], [0.0, 0.0]]}
        with open(self.json_file) as anno_file:
            annotations = json.load(anno_file)

        val_annotation, train_annotation = [], []
        # put to train or val annotation list
        for idx, val in enumerate(annotations):
            # record dataset name
            if self.dataset_name is None:
                self.dataset_name = val['dataset']

            if val['isValidation'] == True:
                val_annotation.append(annotations[idx])
            else:
                train_annotation.append(annotations[idx])
        return train_annotation, val_annotation

    def get_dataset_name(self):
        return str(self.dataset_name)

    def get_dataset_size(self):
        return len(self.annotations)

    def get_color_mean(self):
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        return mean

    def get_annotations(self):
        return self.annotations

    def get_train_annotations(self):
        return self.train_annotations

    def get_val_annotations(self):
        return self.val_annotations

    def __len__(self):
        return len(self.annotations) // self.batch_size

    def __getitem__(self, i):

        self.batch_metainfo = []
        for n, annotation in enumerate(self.annotations[i*self.batch_size:(i+1)*self.batch_size]):
            sample_index = i*self.batch_size + n
            # generate input image and ground truth heatmap
            image, gt_heatmap, meta = self.process_image(sample_index, annotation)

            # in case we got an empty image, bypass the sample
            if image is None:
                continue

            # form up batch data
            self.batch_images[n, :, :, :] = image
            self.batch_heatmaps[n, :, :, :] = gt_heatmap
            self.batch_metainfo.append(meta)

        # need to feed each hg unit the same gt heatmap,
        # so append a num_hgstack list
        out_heatmaps = []
        for m in range(self.num_hgstack):
            out_heatmaps.append(self.batch_heatmaps)

        if self.with_meta:
            return self.batch_images, out_heatmaps, self.batch_metainfo
        else:
            return self.batch_images, out_heatmaps


    def process_image(self, sample_index, annotation):
        imagefile = os.path.join(self.image_path, annotation['img_paths'])
        img = Image.open(imagefile)
        # make sure image is in RGB mode with 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.array(img)
        img.close()

        # record origin image shape, will store
        # in metainfo
        image_shape = image.shape

        # get center, keypoints and scale
        # center, keypoints point format: (x, y)
        center = np.array(annotation['objpos'])
        keypoints = np.array(annotation['joint_self'])
        scale = annotation['scale_provided']

        # adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        rotate_angle = 0
        # real-time data augmentation for training process
        if self.is_train:
            # random horizontal filp
            image, keypoints, center = random_horizontal_flip(image, keypoints, center, matchpoints=self.horizontal_matchpoints, prob=0.5)

            # random vertical filp
            image, keypoints, center = random_vertical_flip(image, keypoints, center, matchpoints=self.vertical_matchpoints, prob=0.5)

            # random adjust brightness
            image = random_brightness(image)

            # random adjust color level
            image = random_chroma(image)

            # random adjust contrast
            image = random_contrast(image)

            # random adjust sharpness
            image = random_sharpness(image)

            # random convert image to grayscale
            image = random_grayscale(image)

            # random do gaussian blur to image
            image = random_blur(image)

            # random do histogram equalization using CLAHE
            image = random_histeq(image)

            # random adjust scale
            scale = scale * np.random.uniform(0.8, 1.2)

            # generate random rotate angle for image and keypoints transform
            rotate_angle = random_rotate_angle(rotate_range=30, prob=0.5)


        #######################################################################################################
        # 2 solutions of input data preprocess, including:
        #     1. crop single object area from origin image
        #     2. apply rotate augment
        #     3. resize to model input size
        #     4. transform gt keypoints to cropped image reference

        ###############################
        # Option 1 (from origin repo):
        # crop out single object area, resize to input size and normalize image
        image = crop_image(image, center, scale, self.input_shape, rotate_angle)

        # transform keypoints to cropped image reference
        transformed_keypoints = transform_keypoints(keypoints, center, scale, self.output_shape, rotate_angle)
        ###############################


        ###############################
        # Option 2:
        # crop out single object area and transform keypoints coordinates to single object reference
        #image, transformed_keypoints = crop_single_object(image, keypoints, center, scale, self.input_shape)

        #if rotate_angle != 0:
            # rotate single object image and keypoints coordinates when augment
            #image, transformed_keypoints = rotate_single_object(image, transformed_keypoints, rotate_angle)

        # convert keypoints to model output reference
        #transformed_keypoints[:, 0:2] = transformed_keypoints[:, 0:2] / OUTPUT_STRIDE
        ###############################

        #######################################################################################################

        # in case we got an empty image, bypass the sample
        if image is None:
            return None, None, None

        # normalize image
        image = normalize_image(image, self.get_color_mean())

        # generate ground truth keypoint heatmap
        gt_heatmap = generate_gt_heatmap(transformed_keypoints, self.output_shape)

        # meta info
        metainfo = {'sample_index': sample_index, 'center': center, 'scale': scale, 'image_shape': image_shape,
                    'pts': keypoints, 'tpts': transformed_keypoints, 'name': imagefile}

        return image, gt_heatmap, metainfo

    def get_keypoint_classes(self):
        return self.class_names

    def on_epoch_end(self):
        if self.is_train:
            # Shuffle dataset for next epoch
            random.shuffle(self.annotations)

