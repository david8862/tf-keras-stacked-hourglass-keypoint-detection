#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random
import numpy as np
from PIL import Image
import json
from common.data_utils import crop_image, horizontal_flip, vertical_flip, normalize_image, transform_keypoints, generate_gtmap

# by default, Stacked Hourglass model use output_stride = 4, which means:
#
#   input image:    256 x 256 x 3
#   output heatmap: 64 x 64 x num_classes
#
HG_OUTPUT_STRIDE = 4


class hourglass_dataset(object):
    def __init__(self, dataset_path, class_names, input_size, is_train, matchpoints=None):
        self.jsonfile = os.path.join(dataset_path, 'annotations.json')
        self.is_train = is_train
        self.imgpath = os.path.join(dataset_path, 'images')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.input_size = input_size
        # output heatmap size should be 1/HG_OUTPUT_STRIDE of input size
        self.output_size = (self.input_size[0]//HG_OUTPUT_STRIDE, self.input_size[1]//HG_OUTPUT_STRIDE)
        self.dataset_name = None
        self.annotations = self._load_image_annotation()
        self.horizontal_matchpoints, self.vertical_matchpoints = self._get_matchpoint_list(matchpoints)

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
        with open(self.jsonfile) as anno_file:
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

        if self.is_train:
            return train_annotation
        else:
            return val_annotation

    def get_dataset_name(self):
        return str(self.dataset_name)

    def get_dataset_size(self):
        return len(self.annotations)

    def get_color_mean(self):
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        return mean

    def get_annotations(self):
        return self.annotations

    def generator(self, batch_size, num_hgstack, sigma=1, with_meta=False, is_shuffle=False,
                  rot_flag=False, scale_flag=False, h_flip_flag=False, v_flip_flag=False):
        '''
        Input:  batch_size * input_size  * channel (3)
        Output: batch_size * output_size * num_classes
        '''
        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            assert (rot_flag == False), 'rot_flag must be off in val model'

        while True:
            if is_shuffle:
                random.shuffle(self.annotations)

            batch_images = np.zeros(shape=(batch_size, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
            batch_heatmaps = np.zeros(shape=(batch_size, self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
            batch_metainfo = list()

            count = 0
            for i, annotation in enumerate(self.annotations):
                # generate input image and ground truth heatmap
                image, gt_heatmap, meta = self.process_image(i, annotation, sigma, rot_flag, scale_flag, h_flip_flag, v_flip_flag)

                # in case we got an empty image, bypass the sample
                if image is None:
                    continue

                index = count % batch_size
                # form up batch data
                batch_images[index, :, :, :] = image
                batch_heatmaps[index, :, :, :] = gt_heatmap
                batch_metainfo.append(meta)
                count = count + 1

                if index == (batch_size - 1):
                    # need to feed each hg unit the same gt heatmap,
                    # so append a num_hgstack list
                    out_heatmaps = []
                    for m in range(num_hgstack):
                        out_heatmaps.append(batch_heatmaps)

                    if with_meta:
                        yield batch_images, out_heatmaps, batch_metainfo
                        batch_metainfo = []
                    else:
                        yield batch_images, out_heatmaps

    def process_image(self, sample_index, annotation, sigma, rot_flag, scale_flag, h_flip_flag, v_flip_flag):
        imagefile = os.path.join(self.imgpath, annotation['img_paths'])
        img = Image.open(imagefile)
        # make sure image is in RGB mode with 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.array(img)
        img.close()

        # get center, joints and scale
        # center, joints point format: (x, y)
        center = np.array(annotation['objpos'])
        joints = np.array(annotation['joint_self'])
        scale = annotation['scale_provided']

        # adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        # random horizontal filp
        if h_flip_flag and random.choice([0, 1]):
            image, joints, center = horizontal_flip(image, joints, center, self.horizontal_matchpoints)

        # random vertical filp
        if v_flip_flag and random.choice([0, 1]):
            image, joints, center = vertical_flip(image, joints, center, self.vertical_matchpoints)

        # random adjust scale
        if scale_flag:
            scale = scale * np.random.uniform(0.8, 1.2)

        # random rotate image
        if rot_flag and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
        else:
            rot = 0

        # crop out single person area, resize to input size res and normalize image
        image = crop_image(image, center, scale, self.input_size, rot)

        # in case we got an empty image, bypass the sample
        if image is None:
            return None, None, None

        # normalize image
        image = normalize_image(image, self.get_color_mean())

        # transform keypoints to cropped image reference
        transformed_keypoints = transform_keypoints(joints, center, scale, self.output_size, rot)
        # generate ground truth point heatmap
        gtmap = generate_gtmap(transformed_keypoints, sigma, self.output_size)

        # meta info
        metainfo = {'sample_index': sample_index, 'center': center, 'scale': scale,
                    'pts': joints, 'tpts': transformed_keypoints, 'name': imagefile}

        return image, gtmap, metainfo

    def get_keypoint_classes(self):
        return self.class_names

