#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import cv2, scipy, copy
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from hourglass.data import hourglass_dataset
from hourglass.postprocess import post_process_heatmap
from common.data_utils import transform
from common.utils import get_classes


def get_predicted_kp_from_htmap(heatmap, meta, outres):
    # nms to get location
    kplst = post_process_heatmap(heatmap)
    kps = np.array(kplst)

    # use meta information to transform back to original image
    mkps = copy.copy(kps)
    for i in range(kps.shape[0]):
        mkps[i, 0:2] = transform(kps[i], meta['center'], meta['scale'], res=outres, invert=1, rot=0)

    return mkps


def get_final_pred_kps(val_keypoints, preheatmap, metainfo, output_size):
    for i in range(preheatmap.shape[0]):
        prehmap = preheatmap[i, :, :, :]
        meta = metainfo[i]
        sample_index = meta['sample_index']
        kps = get_predicted_kp_from_htmap(prehmap, meta, output_size)
        val_keypoints[sample_index, :, :] = kps[:, 0:2]  # ignore the visibility


def get_visible_array(val_annotations):
    visible_list = []
    for val_record in val_annotations:
        visible_record = []

        key_points = val_record['joint_self']
        for key_point in key_points:
            if key_point == [0., 0., 0.]:
                visible_record.append(0)
            else:
                visible_record.append(1)

        visible_list.append(visible_record)

    visible_array = np.array(visible_list, dtype='uint8')
    visible_array = np.transpose(visible_array, [1, 0])
    print('visible_array shape: ', visible_array.shape)
    return visible_array


def get_pos_gt(val_annotations):
    key_point_list = []

    for val_record in val_annotations:
        key_point_record = []

        key_points = val_record['joint_self']
        for key_point in key_points:
            #only pick (x,y) and drop is_visible
            key_point_record.append(key_point[:2])

        key_point_list.append(key_point_record)

    key_point_array = np.array(key_point_list, dtype='float64')
    key_point_array = np.transpose(key_point_array, [1, 2, 0])
    print('key_point_array shape: ', key_point_array.shape)
    return key_point_array


def get_headboxes(val_annotations):
    headboxes_list = []

    for val_record in val_annotations:
        headboxes_record = val_record['headboxes']
        headboxes_list.append(headboxes_record)

    headboxes_array = np.array(headboxes_list, dtype='float64')
    headboxes_array = np.transpose(headboxes_array, [1, 2, 0])
    print('headboxes_array shape: ', headboxes_array.shape)
    return headboxes_array



def eval_pckh(model_name, val_keypoints, val_annotations, class_names, threshold = 0.5):
    SC_BIAS = 0.6

    #dict = loadmat('data/mpii/detections_our_format.mat')
    #jnt_missing = dict['jnt_missing']
    #jnt_visible = 1 - jnt_missing
    #pos_gt_src = dict['pos_gt_src']
    #headboxes_src = dict['headboxes_src']

    jnt_visible = get_visible_array(val_annotations)
    pos_gt_src = get_pos_gt(val_annotations)
    headboxes_src = get_headboxes(val_annotations)

    # predictions
    pos_pred_src = np.transpose(val_keypoints, [1, 2, 0])

    # get index of MPII 16 keypoints
    head_top = class_names.index('head_top')
    left_shoulder = class_names.index('left_shoulder')
    left_elbow = class_names.index('left_elbow')
    left_wrist = class_names.index('left_wrist')
    left_hip = class_names.index('left_hip')
    left_knee = class_names.index('left_knee')
    left_ankle = class_names.index('left_ankle')

    right_shoulder = class_names.index('right_shoulder')
    right_elbow = class_names.index('right_elbow')
    right_wrist = class_names.index('right_wrist')
    right_hip = class_names.index('right_hip')
    right_knee = class_names.index('right_knee')
    right_ankle = class_names.index('right_ankle')

    # calculate PCKh@0.5 score
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    print("Model,  Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean")
    print('{:s}   {:.2f}  {:.2f}     {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(model_name, PCKh[head_top],
                                                                                                0.5 * (PCKh[left_shoulder] +
                                                                                                       PCKh[right_shoulder]) \
                                                                                                , 0.5 * (PCKh[left_elbow] +
                                                                                                         PCKh[right_elbow]),
                                                                                                0.5 * (PCKh[left_wrist] +
                                                                                                       PCKh[right_wrist]),
                                                                                                0.5 * (PCKh[left_hip] +
                                                                                                       PCKh[right_hip]),
                                                                                                0.5 * (PCKh[left_knee] +
                                                                                                       PCKh[right_knee]) \
                                                                                                , 0.5 * (PCKh[left_ankle] +
                                                                                                         PCKh[right_ankle]),
                                                                                                np.mean(PCKh)))

def main(args):
    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)

    # load trained model for eval
    model = load_model(args.model_path, compile=False)

    # get input size, assume only 1 input
    input_size = tuple(model.input.shape.as_list()[1:3])

    # check & get output size
    output_tensor = model.output
    # check to handle multi-output model
    if isinstance(output_tensor, list):
        output_tensor = output_tensor[-1]
    output_size = tuple(output_tensor.shape.as_list()[1:3])

    # check for any invalid input & output size
    assert None not in input_size, 'Invalid input size.'
    assert None not in output_size, 'Invalid output size.'
    assert output_size[0] == input_size[0]//4 and output_size[1] == input_size[1]//4, 'output size should be 1/4 of input size.'

    # prepare validation dataset
    valdata = hourglass_dataset(args.dataset_path, class_names,
                          input_size=input_size, is_train=False)

    print('validation data size', valdata.get_dataset_size())

    # form up the validation result matrix
    val_keypoints = np.zeros(shape=(valdata.get_dataset_size(), num_classes, 2), dtype=np.float)

    count = 0
    batch_size = 8
    val_gen = valdata.generator(batch_size, num_hgstack=1, sigma=1, is_shuffle=False, with_meta=True)
    pbar = tqdm(total=valdata.get_dataset_size(), desc='Eval model')
    # fetch validation data from generator, which will crop out single person area, resize to inres and normalize image
    for _img, _gthmap, _meta in val_gen:
        # get predicted heatmap
        prediction = model.predict(_img)
        if isinstance(prediction, list):
            prediction = prediction[-1]
        # transform predicted heatmap to final keypoint output,
        # and store it into result matrix
        get_final_pred_kps(val_keypoints, prediction, _meta, output_size)

        count += batch_size
        if count > valdata.get_dataset_size():
            break
        pbar.update(batch_size)
    pbar.close()

    # store result matrix, and use it to get PCKh metrics
    eval_pckh(args.model_path, val_keypoints, valdata.get_annotations(), class_names)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate PCKh metric on MPII dataset for Hourglass keypoint detection model')

    parser.add_argument('--model_path', type=str, required=True, help='path to model file')
    parser.add_argument('--dataset_path', type=str, required=False, default='../data/mpii',
        help='dataset path containing images and annotation file, default=../data/mpii')
    parser.add_argument('--classes_path', type=str, required=False, default='../configs/mpii_classes.txt',
        help='path to keypoint class definitions, default=../configs/mpii_classes.txt')

    args = parser.parse_args()

    main(args)

