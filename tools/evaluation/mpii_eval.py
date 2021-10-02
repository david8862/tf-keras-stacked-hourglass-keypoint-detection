#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import cv2, scipy, copy
import numpy as np
from tqdm import tqdm

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import MNN
import onnxruntime

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from hourglass.data import hourglass_dataset
from hourglass.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import invert_transform_keypoints, revert_keypoints
from common.utils import get_classes

from eval import hourglass_predict_keras, hourglass_predict_tflite, hourglass_predict_pb, hourglass_predict_onnx, hourglass_predict_mnn, revert_pred_keypoints, load_eval_model, draw_plot_func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def fill_eval_array(eval_keypoints_array, pred_keypoints, metainfo, model_input_shape, output_shape):
    # get sample index from meta info
    sample_index = metainfo['sample_index']

    # revert predict keypoints back to origin image shape
    reverted_pred_keypoints = revert_pred_keypoints(pred_keypoints, metainfo, model_input_shape, output_shape)

    # fill result array at sample_index
    eval_keypoints_array[sample_index, :, :] = reverted_pred_keypoints[:, 0:2]  # ignore the visibility


def get_visible_array(eval_annotations):
    """
    parse visible array from eval annotation list
    """
    visible_list = []
    for eval_record in eval_annotations:
        visible_record = []

        keypoints = eval_record['joint_self']
        for keypoint in keypoints:
            if keypoint == [0., 0., 0.]:
                visible_record.append(0)
            else:
                visible_record.append(1)

        visible_list.append(visible_record)
    visible_array = np.array(visible_list, dtype='uint8')
    # transpose array for computing PCKh
    visible_array = np.transpose(visible_array, [1, 0])

    #print('visible_array shape: ', visible_array.shape)
    return visible_array


def get_gt_keypoints_array(eval_annotations):
    """
    parse gt keypoints array from eval annotation list
    """
    keypoints_list = []
    for eval_record in eval_annotations:
        keypoints_record = []

        keypoints = eval_record['joint_self']
        for keypoint in keypoints:
            #only pick (x,y) and drop is_visible
            keypoints_record.append(keypoint[:2])

        keypoints_list.append(keypoints_record)
    keypoints_array = np.array(keypoints_list, dtype='float64')
    # transpose array for computing PCKh
    keypoints_array = np.transpose(keypoints_array, [1, 2, 0])

    #print('keypoints_array shape: ', keypoints_array.shape)
    return keypoints_array


def get_headboxes_array(eval_annotations):
    """
    parse headboxes array from eval annotation list
    """
    headboxes_list = []
    for eval_record in eval_annotations:
        headboxes_record = eval_record['headboxes']
        headboxes_list.append(headboxes_record)

    headboxes_array = np.array(headboxes_list, dtype='float64')
    # transpose array for computing PCKh
    headboxes_array = np.transpose(headboxes_array, [1, 2, 0])

    #print('headboxes_array shape: ', headboxes_array.shape)
    return headboxes_array


def eval_PCKh(eval_keypoints_array, eval_annotations, class_names, threshold=0.5):
    SC_BIAS = 0.6

    # parse gt keypoints array, visible array and headboxes array
    # for computing PCKh
    gt_keypoints_array = get_gt_keypoints_array(eval_annotations)
    keypoint_visible_array = get_visible_array(eval_annotations)
    headboxes_array = get_headboxes_array(eval_annotations)

    # transpose eval result array for computing PCKh
    eval_keypoints_array = np.transpose(eval_keypoints_array, [1, 2, 0])

    # keypoints error
    keypoints_error = eval_keypoints_array - gt_keypoints_array
    keypoints_error = np.linalg.norm(keypoints_error, axis=1)

    # normalized head size
    headsizes = headboxes_array[1, :, :] - headboxes_array[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS

    # scaled keypoints error with head size
    scale = np.multiply(headsizes, np.ones((len(keypoints_error), 1)))
    scaled_keypoints_error = np.divide(keypoints_error, scale)
    scaled_keypoints_error = np.multiply(scaled_keypoints_error, keypoint_visible_array)

    # calculate PCKh@threshold score
    keypoint_count = np.sum(keypoint_visible_array, axis=1)
    less_than_threshold = np.multiply((scaled_keypoints_error < threshold), keypoint_visible_array)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), keypoint_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True


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

    # form PCKh metric dict
    pckh_dict = {}
    pckh_dict['Head'] = round(PCKh[head_top], 2)
    pckh_dict['Shoulder'] = round(0.5 * (PCKh[left_shoulder] + PCKh[right_shoulder]), 2)
    pckh_dict['Elbow'] = round(0.5 * (PCKh[left_elbow] + PCKh[right_elbow]), 2)
    pckh_dict['Wrist'] = round(0.5 * (PCKh[left_wrist] + PCKh[right_wrist]), 2)
    pckh_dict['Hip'] = round(0.5 * (PCKh[left_hip] + PCKh[right_hip]), 2)
    pckh_dict['Knee'] = round(0.5 * (PCKh[left_knee] + PCKh[right_knee]), 2)
    pckh_dict['Ankle'] = round(0.5 * (PCKh[left_ankle] + PCKh[right_ankle]), 2)

    # show PCKh metric
    print('\nPCKh evaluation')
    print("Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle, Mean")
    print('{:.2f} {:.2f}  {:.2f}  {:.2f} {:.2f}  {:.2f}  {:.2f} {:.2f}'.format(pckh_dict['Head'],
                                                                               pckh_dict['Shoulder'],
                                                                               pckh_dict['Elbow'],
                                                                               pckh_dict['Wrist'],
                                                                               pckh_dict['Hip'],
                                                                               pckh_dict['Knee'],
                                                                               pckh_dict['Ankle'],
                                                                               np.mean(PCKh)))

    # draw PCKh plot
    window_title = "PCKh evaluation"
    plot_title = "PCKh@{0} mean score = {1:.2f}%".format(threshold, np.mean(PCKh))
    x_label = "Accuracy"
    os.makedirs('result', exist_ok=True)
    output_path = os.path.join('result','PCKh.png')
    draw_plot_func(pckh_dict, len(pckh_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')



def mpii_eval(model, model_format, eval_dataset, class_names, model_input_shape, score_threshold, conf_threshold):
    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    # form up empty eval result array
    eval_keypoints_array = np.zeros(shape=(eval_dataset.get_dataset_size(), len(class_names), 2), dtype=np.float)

    count = 0
    pbar = tqdm(total=eval_dataset.get_dataset_size(), desc='Eval model')
    for image_data, gt_heatmap, metainfo in eval_dataset:
        # fetch validation data from generator, which will crop out single person area, resize to input_shape and normalize image
        count += 1
        if count > eval_dataset.get_dataset_size():
            break

        # support of tflite model
        if model_format == 'TFLITE':
            heatmap = hourglass_predict_tflite(model, image_data)
        # support of MNN model
        elif model_format == 'MNN':
            heatmap = hourglass_predict_mnn(model, session, image_data)
        # support of TF 1.x frozen pb model
        elif model_format == 'PB':
            heatmap = hourglass_predict_pb(model, image_data)
        # support of ONNX model
        elif model_format == 'ONNX':
            heatmap = hourglass_predict_onnx(model, image_data)
        # normal keras h5 model
        elif model_format == 'H5':
            heatmap = hourglass_predict_keras(model, image_data)
        else:
            raise ValueError('invalid model format')

        heatmap_shape = heatmap.shape[0:2]
        metainfo = metainfo[0]

        # get predict keypoints from heatmap
        pred_keypoints = post_process_heatmap_simple(heatmap, conf_threshold)
        pred_keypoints = np.array(pred_keypoints)

        # revert predict keypoints to origin image shape,
        # and fill into eval result array
        fill_eval_array(eval_keypoints_array, pred_keypoints, metainfo, model_input_shape, heatmap_shape)

        pbar.update(1)
    pbar.close()

    # get PCKh metrics with eval result array and gt annotations
    eval_PCKh(eval_keypoints_array, eval_dataset.get_annotations(), class_names, score_threshold)
    return


def main():
    parser = argparse.ArgumentParser(description='Calculate PCKh metric on MPII dataset for keypoint detection model')

    parser.add_argument('--model_path', type=str, required=True, help='path to model file')
    parser.add_argument('--dataset_path', type=str, required=False, default='../../data/mpii',
        help='dataset path containing images and annotation file, default=%(default)s')
    parser.add_argument('--classes_path', type=str, required=False, default='../../configs/mpii_classes.txt',
        help='path to keypoint class definitions, default=%(default)s')
    parser.add_argument('--score_threshold', type=float, required=False, default=0.5,
        help='score threshold for PCK evaluation, default=%(default)s')
    parser.add_argument('--conf_threshold', type=float, required=False, default=1e-6,
        help='confidence threshold for filtering keypoint in postprocess, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='256x256',
        help='model image input shape as <height>x<width>, default=%(default)s')

    args = parser.parse_args()

    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    # load trained model for eval
    model, model_format = load_eval_model(args.model_path)

    # prepare eval dataset
    eval_dataset = hourglass_dataset(args.dataset_path, batch_size=1, class_names=class_names,
                              input_shape=model_input_shape, num_hgstack=1, is_train=False, with_meta=True)
    print('eval data size', eval_dataset.get_dataset_size())

    mpii_eval(model, model_format, eval_dataset, class_names, model_input_shape, args.score_threshold, args.conf_threshold)


if __name__ == '__main__':
    main()

