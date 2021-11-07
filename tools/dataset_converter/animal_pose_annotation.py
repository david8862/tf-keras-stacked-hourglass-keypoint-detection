#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert Animal-Pose keypoint dataset (4608 images, five categories: dog/cat/cow/horse/sheep) annotation to our annotation file

Dataset website:
https://sites.google.com/view/animal-pose/

images and json annotation could be download from:
https://drive.google.com/drive/folders/1xxm6ZjfsDSmv6C9JvbgiGrmHktrUjV5x
"""
import os, sys, argparse
import json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import MPII_SCALE_REFERENCE


def parse_animal_pose_annotation(annotation_path, val_split, category_ids=None):
    animal_pose_annotation_file = open(annotation_path, encoding='utf-8')
    # animal_pose_annotation_data format:
    # {
    #  "info": info,
    #  "images": [image],
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    animal_pose_annotation_data = json.load(animal_pose_annotation_file)
    animal_pose_annotations = animal_pose_annotation_data['annotations']

    # animal_pose_images_dict is mapping of image_id and image file name,
    # with format:
    # {
    #  '1': '2007_000063.jpg',
    #  '2': '2007_000175.jpg',
    #  '3': '2007_000332.jpg',
    #  ...
    # }
    animal_pose_images_dict = animal_pose_annotation_data['images']

    annotation_list = []
    pbar = tqdm(total=len(animal_pose_annotations), desc='Animal Pose annotation')
    for animal_pose_annotation in animal_pose_annotations:
        # animal_pose_annotation format:
        # {
        #  "keypoints": [[x1,y1,v1],...],
        #  "num_keypoints": int,
        #  "image_id": int,
        #  "category_id": int,
        #  "bbox": [xmin,ymin,xmax,ymax],
        # }
        pbar.update(1)
        # if no annotated keypoints, bypass the sample
        num_keypoints = animal_pose_annotation['num_keypoints']
        if num_keypoints <= 0:
            continue

        if category_ids:
            # filter animal sample with a category id list
            category_id = animal_pose_annotation['category_id']
            if category_id not in category_ids:
                continue

        # parse image file name with image_id
        image_id = animal_pose_annotation['image_id']
        image_name = animal_pose_images_dict[str(image_id)]

        # parse keypoints
        keypoints = animal_pose_annotation['keypoints']

        # calculate objpos and scale from object bbox
        bbox = animal_pose_annotation['bbox']
        xmin,ymin,xmax,ymax = bbox

        # center point of bbox as objpos
        objpos = [(xmin+xmax)/2, (ymin+ymax)/2]
        # use longer one in (width, height) to get scale
        scale = float(max(xmax-xmin, ymax-ymin)) / MPII_SCALE_REFERENCE
        if scale <= 0:
            continue

        # assign train/val according to val_split
        is_validation = float(np.random.rand() < val_split)

        #form up annotation dict item
        annotation_record = {}
        annotation_record['dataset'] = 'Animal Pose'
        annotation_record['img_paths'] = image_name
        annotation_record['isValidation'] = is_validation
        annotation_record['joint_self'] = keypoints
        annotation_record['objpos'] = objpos
        annotation_record['scale_provided'] = scale
        annotation_record['headboxes'] = list([[0.0, 0.0], [0.0, 0.0]])

        annotation_list.append(annotation_record)

    pbar.close()
    animal_pose_annotation_file.close()
    print('converted sample number: {}'.format(len(annotation_list)))
    return annotation_list


def parse_animal_pose_keypoint_info(annotation_path, class_path, skeleton_path):
    animal_pose_annotation_file = open(annotation_path, encoding='utf-8')
    # animal_pose_annotation_data format:
    # {
    #  "info": info,
    #  "images": [image],
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    animal_pose_annotation_data = json.load(animal_pose_annotation_file)

    # animal_pose_categories format:
    # {
    #     "supercategory": 'animal',
    #     "id": int,
    #     "name": str,
    #     "keypoints": [str],
    #     "skeleton": [edge]
    # }
    animal_pose_categories = animal_pose_annotation_data['categories'][0]
    # keypoints format: ['left_eye', 'right_eye', 'nose', ... , 'tailbase']
    keypoints = animal_pose_categories['keypoints']
    # skeleton format: [[0,1], [0,2], ... ,[11,15],[12,16]]
    skeletons = animal_pose_categories['skeleton']

    # save keypoint class names
    class_file = open(class_path, 'w')
    for keypoint in keypoints:
        class_file.write(keypoint)
        class_file.write('\n')
    class_file.close()

    # save skeleton definitions
    skeleton_file = open(skeleton_path, 'w')
    for skeleton in skeletons:
        # skeleton line format: [start_point_name,end_point_name,color]
        skeleton_file.write(keypoints[skeleton[0]])
        skeleton_file.write(',')
        skeleton_file.write(keypoints[skeleton[1]])
        # by default use red skeleton, can manually change
        skeleton_file.write(',r\n')
    skeleton_file.close()

    animal_pose_annotation_file.close()


def get_category_ids(annotation_path, animal_names):
    animal_pose_annotation_file = open(annotation_path, encoding='utf-8')
    # animal_pose_annotation_data format:
    # {
    #  "info": info,
    #  "images": [image],
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    animal_pose_annotation_data = json.load(animal_pose_annotation_file)

    # animal_pose_categories is category dict list with format:
    # {
    #     "supercategory": 'animal',
    #     "id": int,
    #     "name": str,
    #     "keypoints": [str],
    #     "skeleton": [edge]
    # }
    animal_pose_categories = animal_pose_annotation_data['categories']

    category_ids = []
    for animal_pose_category in animal_pose_categories:
        # found animal name in categories and record category id in list
        if animal_pose_category['name'] in animal_names:
            category_ids.append(animal_pose_category['id'])

    animal_pose_annotation_file.close()
    return category_ids



def animal_pose_annotation(args):
    # if specify animal name, will generate a category id list to filter
    # corresponding samples from dataset
    if args.animal_names:
        animal_names = args.animal_names.split(',')
        category_ids = get_category_ids(args.annotation_path, animal_names)
        if len(category_ids) == 0:
            category_ids = None
    else:
        category_ids = None

    # parse animal pose annotations and save to our json annotation
    annotation_list = parse_animal_pose_annotation(args.annotation_path, args.val_split, category_ids)

    f = open(args.output_anno_path, 'w')
    json.dump(annotation_list, f)

    # parse animal pose keypoints & skeleton from annotation
    parse_animal_pose_keypoint_info(args.annotation_path, args.output_class_path, args.output_skeleton_path)
    return


def main():
    parser = argparse.ArgumentParser(description='Parse Animal-Pose keypoint annotation to our annotation files')
    parser.add_argument('--annotation_path', type=str, required=True, help='Animal-Pose keypoint annotation file path')
    parser.add_argument('--animal_names', type=str, required=False, help='selected animals in (dog/cat/cow/horse/sheep), separate with comma. Will involve all if None. default=%(default)s', default=None)
    parser.add_argument('--val_split', type=float, required=False, help='validation data persentage in dataset, default=%(default)s', default=0.1)
    parser.add_argument('--output_anno_path', type=str, required=False,  help='generated annotation json file path, default=%(default)s', default='./annotations.json')
    parser.add_argument('--output_class_path', type=str, required=False,  help='generated keypoint classes txt file path, default=%(default)s', default='./animal_pose_classes.txt')
    parser.add_argument('--output_skeleton_path', type=str, required=False,  help='generated keypoint skeleton txt file path, default=%(default)s', default='./animal_pose_skeleton.txt')
    args = parser.parse_args()

    # Animal-Pose keypoint annotation follow MSCOCO json format,
    # but with some different details
    animal_pose_annotation(args)

if __name__ == '__main__':
    main()
