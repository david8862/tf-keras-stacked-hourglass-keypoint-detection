#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare person keypoints with OKS or PCK metric
"""
import os, sys, argparse, json
import math
#import numpy as np


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_square_box(box):
    '''expand person bbox to square'''
    xmin, ymin, xmax, ymax = map(int, box)

    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    length = max(xmax-xmin, ymax-ymin)

    square_xmin = center_x - length//2
    square_xmax = center_x + length//2
    square_ymin = center_y - length//2
    square_ymax = center_y + length//2

    return square_xmin, square_ymin, square_xmax, square_ymax


def get_normalize_scale(normalize_shape=(256, 256)):
    """
    rescale keypoint distance normalize coefficient
    based on area shape, used for PCK evaluation

    NOTE: "6.4*4" is standard normalize coefficient under
          shape (256,256)

    # Arguments
        normalize_shape: normalized person area shape as (height, width)

    # Returns
        scale: normalize coefficient
    """
    # use averaged scale factor if non square input shape
    scale = float((normalize_shape[0] + normalize_shape[1]) / 2) / 256.0

    return 6.4 * 4 * scale


def check_pred_keypoints(pred_keypoint, gt_keypoint, threshold, normalize_scale):
    # check if ground truth keypoint is valid
    if gt_keypoint[0] >= 0 and gt_keypoint[1] >= 0:
        # calculate normalized euclidean distance between pred and gt keypoints
        dx = gt_keypoint[0] - pred_keypoint[0]
        dy = gt_keypoint[1] - pred_keypoint[1]
        distance = math.sqrt(dx**2 + dy**2) / normalize_scale

        if distance < threshold:
            # succeed prediction
            return 1
        else:
            # fail prediction
            return 0
    else:
        # invalid gt keypoint
        return 0


def keypoint_matching(pred_keypoints, gt_keypoints, threshold, normalize_scale):
    assert len(pred_keypoints) == len(gt_keypoints), 'keypoint number mismatch'

    matching_list = []
    for i in range(len(gt_keypoints)):
        # compare pred keypoint with gt keypoint to get result
        result = check_pred_keypoints(pred_keypoints[i], gt_keypoints[i], threshold, normalize_scale)
        matching_list.append(result)

    return matching_list


def pck_similarity(pred_keypoints, gt_keypoints, normalize_shape, distance_threshold):
    """
    Calculate PCK (Percentage of Correct Keypoints) for single object keypoints

    # Arguments
        pred_keypoints: normalized predict keypoints coordinate,
            list of points with shape (num_keypoints, 2),
        gt_keypoints: normalized ground truth keypoints coordinate,
            list of points with shape (num_keypoints, 2),
        normalize_shape: normalized person area shape,
            tuple with (height, width) format
        distance_threshold: normalized distance threshold for keypoints compare,
            float number

    # Returns
        pck: PCK value.
    """
    normalize_scale = get_normalize_scale(normalize_shape)

    # calculate succeed & failed keypoints for prediction
    matching_list = keypoint_matching(pred_keypoints, gt_keypoints, distance_threshold, normalize_scale)

    # correct percentage
    accuracy = sum(matching_list) * 1.0 / len(matching_list)

    return accuracy


def sign(x):
    """
    Get signal of a number, like np.sign()
    """
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        raise ValueError('invalid data value!')


def oks_similarity(pred_keypoints, gt_keypoints, normalize_shape, class_names):
    """
    Calculate OKS (Object Keypoint Similarity) for single object keypoints

    # Arguments
        pred_keypoints: normalized predict keypoints coordinate,
            list of points with shape (num_keypoints, 2),
        gt_keypoints: normalized ground truth keypoints coordinate,
            list of points with shape (num_keypoints, 2),
        normalize_shape: normalized person area shape,
            tuple with (height, width) format

    # Returns
        oks: OKS value.
    """
    assert len(pred_keypoints) == len(gt_keypoints), 'keypoint number mismatch'

    # generate variance list
    if len(gt_keypoints) == 16:
        # MPII 16 keypoints
        sigmas = [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
        variance = []
        for sigma in sigmas:
            variance.append((sigma*2)**2)
    elif len(gt_keypoints) == 17:
        # MSCOCO 17 keypoints
        sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
        variance = []
        for sigma in sigmas:
            variance.append((sigma*2)**2)
    else:
        raise ValueError('invalid keypoints number!')

    # get person bbox area size
    box_area = normalize_shape[0] * normalize_shape[1]

    similarity = 0
    valid_count = 0
    distance_dict = dict()
    for i in range(len(gt_keypoints)):
        # calculate normalized distance for each point
        dx = pred_keypoints[i][0] - gt_keypoints[i][0]
        dy = pred_keypoints[i][1] - gt_keypoints[i][1]
        distance = (dx**2 + dy**2) / (variance[i] * (box_area + 2.22e-16) * 2)

        # use sign(dy) as distance signal, since we mainly judge person action on
        # y axis, like "arm up/down", "leg up/down"
        distance_dict[class_names[i]] = distance * sign(dy)

        # get similarity with exp(-distance), and only count valid ground truth keypoint
        if gt_keypoints[i][0] > 0 and gt_keypoints[i][1] > 0:
            similarity += math.exp(-distance)
            valid_count += 1

    # average similarity
    similarity = similarity / valid_count if valid_count != 0 else 0

    return similarity, distance_dict


def need_to_check(shapes, class_name):
    """
    See if a keypoint in annotate shapes need to be checked
    """
    for shape in shapes:
        if shape['label'] == class_name and len(shape['flags']) > 0:
            return shape['flags']['check']

    return False


def person_action_check(distance_dict, annotate_data):
    # here we design to check only on critical frame
    if len(annotate_data['flags']) == 0 or annotate_data['flags']['critical'] == False:
        return

    shapes = annotate_data['shapes']
    # check left_elbow for left arm
    if need_to_check(shapes, 'left_elbow') and abs(distance_dict['left_elbow']) > 0.3:
        signal = sign(distance_dict['left_elbow'])
        if signal == 1:
            print('left arm up!')
        else:
            print('left arm down!')

    # check right_elbow for right arm
    if need_to_check(shapes, 'right_elbow') and abs(distance_dict['right_elbow']) > 0.3:
        signal = sign(distance_dict['right_elbow'])
        if signal == 1:
            print('right arm up!')
        else:
            print('right arm down!')

    # check left_wrist for left hand
    if need_to_check(shapes, 'left_wrist') and abs(distance_dict['left_wrist']) > 0.3:
        signal = sign(distance_dict['left_wrist'])
        if signal == 1:
            print('left hand up!')
        else:
            print('left hand down!')

    # check right_wrist for right hand
    if need_to_check(shapes, 'right_wrist') and abs(distance_dict['right_wrist']) > 0.3:
        signal = sign(distance_dict['right_wrist'])
        if signal == 1:
            print('right hand up!')
        else:
            print('right hand down!')

    # check left_knee for left leg
    if need_to_check(shapes, 'left_knee') and abs(distance_dict['left_knee']) > 0.3:
        signal = sign(distance_dict['left_knee'])
        if signal == 1:
            print('left leg up!')
        else:
            print('left leg down!')

    # check right_knee for right leg
    if need_to_check(shapes, 'right_knee') and abs(distance_dict['right_knee']) > 0.3:
        signal = sign(distance_dict['right_knee'])
        if signal == 1:
            print('right leg up!')
        else:
            print('right leg down!')

    # check left_ankle for left foot
    if need_to_check(shapes, 'left_ankle') and abs(distance_dict['left_ankle']) > 0.3:
        signal = sign(distance_dict['left_ankle'])
        if signal == 1:
            print('left foot up!')
        else:
            print('left foot down!')

    # check right_ankle for right foot
    if need_to_check(shapes, 'right_ankle') and abs(distance_dict['right_ankle']) > 0.3:
        signal = sign(distance_dict['right_ankle'])
        if signal == 1:
            print('right foot up!')
        else:
            print('right foot down!')


def get_normalize_keypoints(keypoints, normalize_shape=(256, 256)):
    """
    Convert raw keypoints coordinate to normalized person area reference

    # Arguments
        keypoints: raw keypoints coordinate,
            list of points with shape (num_keypoints, 2),
        normalize_shape: normalized person area shape,
            tuple with (height, width) format

    # Returns
        keypoints: normalized keypoints coordinate,
            list of points with shape (num_keypoints, 2),
    """
    # for better alignment, we need to use boundary of keypoints
    # as raw person area bbox
    x_list = [keypoint[0] for keypoint in keypoints]
    y_list = [keypoint[1] for keypoint in keypoints]

    raw_xmin = min(x_list) - 1
    raw_ymin = min(y_list) - 1
    raw_xmax = max(x_list) + 1
    raw_ymax = max(y_list) + 1
    box = [raw_xmin, raw_ymin, raw_xmax, raw_ymax]

    # expand person area bbox to square for normalize
    xmin, ymin, xmax, ymax = get_square_box(box)

    # calculate actual resize ratio on width and height
    square_height = ymax - ymin
    square_width = xmax - xmin
    resize_ratio_x = normalize_shape[1] / float(square_width)
    resize_ratio_y = normalize_shape[0] / float(square_height)

    for i in range(len(keypoints)):
        # only pick valid keypoint
        if keypoints[i][0] > 0 and keypoints[i][1] > 0:
            # move and resize the keypoint to normalize_shape
            normalized_x = (keypoints[i][0] - xmin) * resize_ratio_x
            normalized_y = (keypoints[i][1] - ymin) * resize_ratio_y

            # only pick valid new keypoint
            if normalized_x > 0 and normalized_y > 0:
                keypoints[i] = [normalized_x, normalized_y]

    return keypoints


def get_raw_keypoints(json_data, class_names):
    """
    Parse raw keypoints from labelme format json data

    # Arguments
        json_data: labelme json format keypoints info,
            python dict with labelme struct, see related doc
        class_names: keypoint class name list,
            list of string with keypoints name

    # Returns
        keypoints: raw keypoints coordinate,
            list of points with shape (num_keypoints, 2),
    """
    # init empty keypoints list
    keypoints = []

    class_count = 0
    for i in range(len(class_names)):
        for shape in json_data['shapes']:
            if shape['label'] == class_names[i]:
                # fill keypoints with raw coordinate
                keypoints.append(shape['points'][0])
                class_count += 1

    assert class_count == len(class_names), 'keypoint number mismatch'
    return keypoints


def check_person_num(json_data):
    count = 0
    for shape in json_data['shapes']:
        if shape['label'] == 'person':
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Compare annotate & detect person keypoints similarity from labelme json file')
    parser.add_argument('--annotate_json', help='annotate json file', type=str, required=True)
    parser.add_argument('--detect_json', help='detected result json file', type=str, required=True)
    parser.add_argument('--classes_path', help='path to keypoint class definitions, default=%(default)s', type=str, required=False, default='../configs/mpii_classes.txt')
    parser.add_argument('--normalize_shape', help='normalized person area shape as <height>x<width>, default=%(default)s', type=str, default='256x256')
    parser.add_argument('--metric_type', help="keypoints compare metric type (OKS/PCK), default=%(default)s", type=str, required=False, default='OKS', choices=['OKS', 'PCK'])
    parser.add_argument('--distance_threshold', help='normalized distance threshold for PCK evaluation, default=%(default)s', type=float, required=False, default=0.5)

    args = parser.parse_args()

    # param parse
    class_names = get_classes(args.classes_path)
    height, width = args.normalize_shape.split('x')
    normalize_shape = (int(height), int(width))

    # load json data
    with open(args.annotate_json) as anno_file:
        annotate_data = json.load(anno_file)
    with open(args.detect_json) as det_file:
        detect_data = json.load(det_file)

    # check person number in json data
    if check_person_num(annotate_data) > 1 or check_person_num(detect_data) > 1:
        raise ValueError('invalid json data with more than 1 person!')

    # parse raw keypoints from json data
    gt_keypoints = get_raw_keypoints(annotate_data, class_names)
    pred_keypoints = get_raw_keypoints(detect_data, class_names)

    # convert to normalized keypoints for calculating
    gt_keypoints = get_normalize_keypoints(gt_keypoints, normalize_shape)
    pred_keypoints = get_normalize_keypoints(pred_keypoints, normalize_shape)

    # calculate OKS or PCK metric
    if args.metric_type == 'OKS':
        score, distance_dict = oks_similarity(pred_keypoints, gt_keypoints, normalize_shape, class_names)
        print('Using OKS metric, compare score is:', score)
        print('Keypoint distance:')
        for distance in distance_dict.items():
            print(distance)

        if score > 0.9:
            print('perfect action')
        elif score > 0.75:
            # check person action with keypoints distance, and
            # show related UI message
            person_action_check(distance_dict, annotate_data)
        else:
            print('action mismatch')

    elif args.metric_type == 'PCK':
        score = pck_similarity(pred_keypoints, gt_keypoints, normalize_shape, args.distance_threshold)
        print('Using PCK metric, compare score is:', score)
    else:
        raise ValueError('unsupported metric type!')



if __name__ == '__main__':
    main()
