#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
import numpy as np
from tqdm import tqdm


def parse_coco_keypoints(coco_keypoint_list, keypoint_num):
    """
    Parse keypoints info in MSCOCO annotation

    MSCOCO keypoints format: [x1,y1,type1,x2,y2,type2,...]

    output keypoints format: [[x1+1,y1+1,type1], [x2+1,y2+1,type2],...]
    """
    assert len(coco_keypoint_list) == keypoint_num * 3, 'keypoint number mismatch'

    keypoint_list = []
    for i in range(keypoint_num):
        idx = i * 3
        keypoint = [float(coco_keypoint_list[idx]+1), float(coco_keypoint_list[idx+1]+1), float(coco_keypoint_list[idx+2])]
        keypoint_list.append(keypoint)

    return keypoint_list


def get_objpos(keypoints):
    pos_list = []
    for keypoint in keypoints:
        # keypoint format: [x, y, type]
        # type: 0 - not annotated
        #       1 - annotated but invisible
        #       2 - annotated and visible
        # objpos only count type 1&2 keypoint
        if keypoint[2] > 0:
            pos_list.append(keypoint[0:2])

    objpos = list(np.mean(pos_list, axis=0))
    return objpos


def get_scale(keypoints, keypoint_classes):
    """
    Calculate MPII style scale param on MSCOCO keypoint annotation,
    using mean distance on MPII dataset keypoint pairs to normalize
    corresponding MSCOCO result

    Port from:
    https://github.com/bearpaw/pytorch-pose/blob/master/miscs/cocoScale.m
    """
    pair_names = [
                  ['right_hip', 'right_shoulder'],
                  ['left_shoulder', 'left_hip'],
                  ['right_ankle', 'right_knee'],
                  ['right_knee', 'right_hip'],
                  ['left_hip', 'left_knee'],
                  ['left_knee', 'left_ankle'],
                  ['right_wrist', 'right_elbow'],
                  ['right_elbow', 'right_shoulder'],
                  ['left_shoulder', 'left_elbow'],
                  ['left_elbow', 'left_wrist'],
                 ]

    # Mean distance on MPII dataset
    # rtorso,  ltorso, rlleg, ruleg, lulleg, llleg,
    # rlarm, ruarm, luarm, llarm, head
    mean_distance = [59.3535, 60.4532, 52.1800, 53.7957, 54.4153, 58.0402,
                     27.0043, 32.8498, 33.1757, 27.0978, 33.3005]

    # pair_list value should be
    # [
    #  [13, 7], [6, 12], [17, 15], [15, 13], [12, 14],
    #  [14, 16], [11, 9], [9, 7], [6, 8], [8, 10]
    # ]
    pair_list = [[keypoint_classes.index(pair_name[0]), keypoint_classes.index(pair_name[1])] for pair_name in pair_names]

    scale = -1
    for i, pair in enumerate(pair_list):
        # just use valid keypoints
        if keypoints[pair[0]][2] > 0 and keypoints[pair[1]][2] > 0:
            scale = np.linalg.norm(np.array(keypoints[pair[0]][0:2]) - np.array(keypoints[pair[1]][0:2])) / mean_distance[i]
            break

    return scale


def parse_coco_annotation(annotation_path, annotation_type):
    # check annotation type
    if annotation_type == 'train':
        is_validation = 0.0
    elif annotation_type == 'val':
        is_validation = 1.0
    else:
        raise ValueError('invalid annotation type')

    coco_annotation_file = open(annotation_path, encoding='utf-8')
    # coco_annotation_data format:
    # {
    #  "info": info,
    #  "licenses": [license],
    #  "images": [image],
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    coco_annotation_data = json.load(coco_annotation_file)
    coco_annotations = coco_annotation_data['annotations']

    # coco_categories format:
    # {
    #     "id": int,
    #     "name": str,
    #     "supercategory": str,
    #     "keypoints": [str],
    #     "skeleton": [edge]
    # }
    coco_categories = coco_annotation_data['categories'][0]
    keypoint_classes = coco_categories['keypoints']
    keypoint_num = len(keypoint_classes)

    annotation_list = []
    pbar = tqdm(total=len(coco_annotations), desc='COCO {} annotation'.format(annotation_type))
    for coco_annotation in coco_annotations:
        # coco_annotation format:
        # {
        #  "keypoints": [x1,y1,v1,...],
        #  "num_keypoints": int,
        #  "id": int,
        #  "image_id": int,
        #  "category_id": int,
        #  "segmentation": RLE or [polygon],
        #  "area": float,
        #  "bbox": [x,y,width,height],
        #  "iscrowd": 0 or 1
        # }

        pbar.update(1)
        # if no annotated keypoints, bypass the sample
        num_keypoints = coco_annotation['num_keypoints']
        if num_keypoints <= 0:
            continue

        # person category should be 1
        category_id = coco_annotation['category_id']
        assert category_id == 1, 'invalid category id'

        image_id = coco_annotation['image_id']
        image_name = '%012d.jpg' % (image_id)

        # parse keypoints and get objpos
        keypoints = parse_coco_keypoints(coco_annotation['keypoints'], keypoint_num)
        objpos = get_objpos(keypoints)

        # calculate single person scale from keypoints
        # if couldn't get valid scale, bypass the sample
        scale = get_scale(keypoints, keypoint_classes)
        if scale <= 0:
            continue

        #form up annotation dict item
        annotation_record = {}
        annotation_record['dataset'] = 'COCO'
        annotation_record['img_paths'] = image_name
        annotation_record['isValidation'] = is_validation
        annotation_record['joint_self'] = keypoints
        annotation_record['objpos'] = objpos
        annotation_record['scale_provided'] = scale
        annotation_record['headboxes'] = list([[0.0, 0.0], [0.0, 0.0]])

        annotation_list.append(annotation_record)

    pbar.close()
    coco_annotation_file.close()
    print('converted sample number: {}'.format(len(annotation_list)))
    return annotation_list


def parse_coco_keypoint_info(annotation_path, class_path, skeleton_path):
    coco_annotation_file = open(annotation_path, encoding='utf-8')
    # coco_annotation_data format:
    # {
    #  "info": info,
    #  "licenses": [license],
    #  "images": [image],
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    coco_annotation_data = json.load(coco_annotation_file)

    # coco_categories format:
    # {
    #     "id": int,
    #     "name": str,
    #     "supercategory": str,
    #     "keypoints": [str],
    #     "skeleton": [edge]
    # }
    coco_categories = coco_annotation_data['categories'][0]
    # keypoints format: ["nose","left_eye", ... ,"right_ankle"]
    keypoints = coco_categories['keypoints']
    # skeleton format: [[16,14],[14,12], ... ,[4,6],[5,7]]
    skeletons = coco_categories['skeleton']

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
        skeleton_file.write(keypoints[skeleton[0]-1])
        skeleton_file.write(',')
        skeleton_file.write(keypoints[skeleton[1]-1])
        # by default use red skeleton, can manually change
        skeleton_file.write(',r\n')
    skeleton_file.close()

    coco_annotation_file.close()



def coco_annotation(args):
    # parse coco train/val annotations and save to our json annotation
    train_annotation_list = parse_coco_annotation(args.train_anno_path, 'train')
    val_annotation_list = parse_coco_annotation(args.val_anno_path, 'val')
    annotation_list = val_annotation_list + train_annotation_list

    f = open(args.output_anno_path, 'w')
    json.dump(annotation_list, f)

    # parse coco keypoints&skeleton from val annotation
    parse_coco_keypoint_info(args.val_anno_path, args.output_class_path, args.output_skeleton_path)

    return


def main():
    parser = argparse.ArgumentParser(description='Parse MSCOCO keypoint annotation to our annotation files')
    parser.add_argument('--train_anno_path', type=str, required=True, help='MSCOCO keypoint train annotation file path')
    parser.add_argument('--val_anno_path', type=str, required=True, help='MSCOCO keypoint val annotation file path')
    parser.add_argument('--output_anno_path', type=str, required=False,  help='generated annotation json file path, default=%(default)s', default='./annotations.json')
    parser.add_argument('--output_class_path', type=str, required=False,  help='generated keypoint classes txt file path, default=%(default)s', default='./coco_classes.txt')
    parser.add_argument('--output_skeleton_path', type=str, required=False,  help='generated keypoint skeleton txt file path, default=%(default)s', default='./coco_skeleton.txt')
    args = parser.parse_args()

    coco_annotation(args)

if __name__ == '__main__':
    main()
