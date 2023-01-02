#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
from PIL import Image
import cv2
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_skeleton, render_skeleton
from common.data_utils import MPII_SCALE_REFERENCE
from tools.dataset_converter.coco_annotation import get_objpos


def render_other_people(image, annotation, class_names, skeleton_lines):
    num_other_people = annotation['numOtherPeople']

    # check and get other people info in list
    if num_other_people == 1:
        center_list = np.array([annotation['objpos_other']])
        keypoints_list = np.array([annotation['joint_others']])
        scale_list = [annotation['scale_provided_other']]
    elif num_other_people > 1:
        center_list = np.array(annotation['objpos_other'])
        keypoints_list = np.array(annotation['joint_others'])
        scale_list = annotation['scale_provided_other']
    else:
        return image

    # render one by one
    for i in range(int(num_other_people)):
        center = center_list[i]
        keypoints = keypoints_list[i]
        scale = scale_list[i]

        # form keypoints dict, here we try to show the
        # invisible keypoints
        keypoints_dict = dict()
        for j, keypoint in enumerate(keypoints):
            keypoints_dict[class_names[j]] = (keypoint[0], keypoint[1], 1.0)

        # render keypoints and skeleton lines on image, here we use white line
        image = render_skeleton(image, keypoints_dict, skeleton_lines, colors=(255, 255, 255))

        # draw obj bbox with "center" and "scale" annotation in white
        height, width, channels = image.shape
        obj_size = scale * MPII_SCALE_REFERENCE
        # obj bbox
        xmin = int(max(0, center[0] - (obj_size // 2)))
        xmax = int(min(width, center[0] + (obj_size // 2)))
        ymin = int(max(0, center[1] - (obj_size // 2)))
        ymax = int(min(height, center[1] + (obj_size // 2)))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # obj pos point
        cv2.circle(image, center=(int(center[0]), int(center[1])), color=(255, 255, 255), radius=7, thickness=-1)

    return image


def dataset_visualize(dataset_path, class_names, skeleton_lines):
    json_file = os.path.join(dataset_path, 'annotations.json')
    image_path = os.path.join(dataset_path, 'images')

    with open(json_file) as anno_file:
        annotations = json.load(anno_file)

    print('number of samples:', len(annotations))

    i=0
    while i < len(annotations):
        annotation = annotations[i]

        # load image file
        imagefile = os.path.join(image_path, annotation['img_paths'])
        image = Image.open(imagefile).convert('RGB')
        image = np.array(image, dtype='uint8')

        # get center, keypoints and scale
        # center, keypoints point format: (x, y)
        center = np.array(annotation['objpos'])
        keypoints = np.array(annotation['joint_self'])
        scale = annotation['scale_provided']

        # form keypoints dict, here we try to show the
        # invisible keypoints
        keypoints_dict = dict()
        for j, keypoint in enumerate(keypoints):
            keypoints_dict[class_names[j]] = (keypoint[0], keypoint[1], 1.0)

        # render keypoints and skeleton lines on image
        image = render_skeleton(image, keypoints_dict, skeleton_lines)

        # draw obj bbox with "center" and "scale" annotation
        height, width, channels = image.shape
        obj_size = scale * MPII_SCALE_REFERENCE
        # obj bbox
        xmin = int(max(0, center[0] - (obj_size // 2)))
        xmax = int(min(width, center[0] + (obj_size // 2)))
        ymin = int(max(0, center[1] - (obj_size // 2)))
        ymax = int(min(height, center[1] + (obj_size // 2)))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # obj pos point
        cv2.circle(image, center=(int(center[0]), int(center[1])), color=(255, 0, 0), radius=7, thickness=-1)

        # draw average keypoints with green
        average_keypoints = tuple(map(int, get_objpos(keypoints)))
        cv2.circle(image, center=average_keypoints, color=(0, 255, 0), radius=7, thickness=-1)

        # if have valid head bbox info, show out
        if 'headboxes' in annotation.keys():
            head_xmin = int(max(0, annotation['headboxes'][0][0]))
            head_ymin = int(max(0, annotation['headboxes'][0][1]))
            head_xmax = int(min(width, annotation['headboxes'][1][0]))
            head_ymax = int(min(height, annotation['headboxes'][1][1]))
            cv2.rectangle(image, (head_xmin, head_ymin), (head_xmax, head_ymax), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # if there's other people info in annotation, show them in white
        if 'numOtherPeople' in annotation.keys():
            image = render_other_people(image, annotation, class_names, skeleton_lines)

        # check if a validation sample
        if annotation['isValidation'] == 1.0:
            val_label = ' validate sample'
        else:
            val_label = ''

        # show image file info
        image_file_name = os.path.basename(imagefile)
        cv2.putText(image, image_file_name+'({}/{})'.format(i+1, len(annotations))+val_label,
                    (3, 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        # convert to BGR for cv2.imshow
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        try:
            cv2.namedWindow("Dataset visualize f: forward; b: back; q: quit", 0)
            cv2.imshow("Dataset visualize f: forward; b: back; q: quit", image)
        except Exception as e:
            #print(repr(e))
            print('invalid image', image_path)
            try:
                cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE)
            except Exception as e:
                print('No valid window yet, try next image')
                i = i + 1

        keycode = cv2.waitKey(0) & 0xFF
        if keycode == ord('f'):
            #print('forward to next image')
            if i < len(annotations) - 1:
                i = i + 1
        elif keycode == ord('b'):
            #print('back to previous image')
            if i > 0:
                i = i - 1
        elif keycode == ord('q') or keycode == 27: # 27 is keycode for Esc
            print('exit')
            exit()
        else:
            print('unsupport key')



def main():
    parser = argparse.ArgumentParser(description='visualize dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing images and annotation file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to keypoint class definitions')
    parser.add_argument('--skeleton_path', type=str, required=False, help='path to keypoint skeleton definitions, default=%(default)s', default=None)

    args = parser.parse_args()

    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None
    class_names = get_classes(args.classes_path)

    dataset_visualize(args.dataset_path, class_names, skeleton_lines)


if __name__ == '__main__':
    main()
