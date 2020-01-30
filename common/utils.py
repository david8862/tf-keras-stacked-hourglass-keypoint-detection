#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Miscellaneous utility functions."""

#from PIL import Image
import numpy as np
import os, cv2, colorsys
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import tensorflow as tf


def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
                    #tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
        session = tf.Session(config=config)

        # set session
        K.set_session(session)


def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_skeleton(skeleton_path):
    '''loads the skeleton'''
    with open(skeleton_path) as f:
        skeleton_lines = f.readlines()
    skeleton_lines = [s.strip() for s in skeleton_lines]
    return skeleton_lines


def get_matchpoints(matchpoint_path):
    '''loads the matched keypoints'''
    with open(matchpoint_path) as f:
        matchpoint_lines = f.readlines()
    matchpoint_lines = [s.strip() for s in matchpoint_lines]
    return matchpoint_lines


def render_skeleton(image, keypoints_dict, skeleton_lines=None, conf_threshold=0.1, colors=None):
    """
    Render keypoints skeleton on provided image with
    keypoints dict and skeleton lines definition.
    If no skeleton_lines provided, we'll only render
    keypoints.
    """
    def get_color(color_pattern):
        color = (255, 0, 0)

        if color_pattern == 'r':
            color = (255, 0, 0)
        elif color_pattern == 'g':
            color = (0, 255, 0)
        elif color_pattern == 'b':
            color = (0, 0, 255)
        else:
            raise ValueError('invalid color pattern')

        return color

    def draw_line(img, start_point, end_point, color=(255, 0, 0)):
        x_start, y_start, conf_start = start_point
        x_end, y_end, conf_end = end_point

        if (x_start > 1 and y_start > 1 and conf_start > conf_threshold) and (x_end > 1 and y_end > 1 and conf_end > conf_threshold):
            cv2.circle(img, center=(int(x_start), int(y_start)), color=color, radius=3, thickness=-1)
            cv2.circle(img, center=(int(x_end), int(y_end)), color=color, radius=3, thickness=-1)
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color=color, thickness=1)
        return img

    def draw_keypoints(img, key_points, color):
        for key_point in key_points:
            x, y, conf = key_point
            if x > 1 and y > 1 and conf > conf_threshold:
                cv2.circle(img, center=(int(x), int(y)), color=color, radius=3, thickness=-1)
        return img

    if skeleton_lines:
        for skeleton_line in skeleton_lines:
            #skeleton line format: [start_point_name,end_point_name,color]
            skeleton_list = skeleton_line.split(',')
            color = colors
            if color is None:
                color = get_color(skeleton_list[2])
            image = draw_line(image, keypoints_dict[skeleton_list[0]], keypoints_dict[skeleton_list[1]], color=color)
    else:
        if colors is None:
            colors = (0, 0, 0)
        image = draw_keypoints(image, list(keypoints_dict.values()), colors)

    return image

