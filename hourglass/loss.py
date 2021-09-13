#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error#, huber
import tensorflow.keras.backend as K


def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))


def weighted_mse_loss(y_true, y_pred):
    """
    apply weights on heatmap mse loss to only pick valid keypoint heatmap

    since y_true would be gt_heatmap with shape
    (batch_size, heatmap_size[0], heatmap_size[1], num_keypoints)
    we sum up the heatmap for each keypoints and check. Sum for invalid
    keypoint would be 0, so we can get a keypoint weights tensor with shape
    (batch_size, 1, 1, num_keypoints)
    and multiply to loss

    """
    heatmap_sum = K.sum(K.sum(y_true, axis=1, keepdims=True), axis=2, keepdims=True)

    # keypoint_weights shape: (batch_size, 1, 1, num_keypoints), with
    # valid_keypoint = 1.0, invalid_keypoint = 0.0
    keypoint_weights = 1.0 - K.cast(K.equal(heatmap_sum, 0.0), 'float32')

    return K.sqrt(K.mean(K.square((y_true - y_pred) * keypoint_weights)))


def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return loss


def huber_loss(y_true, y_pred, clip_delta=1.0):
    diff = K.abs(y_true - y_pred)
    less_than_delta = K.cast(K.less(diff, clip_delta), 'float32')
    loss = (less_than_delta * 0.5 * diff**2) + (1 - less_than_delta) * (clip_delta * diff - 0.5 * (clip_delta**2))

    return loss


def get_loss(loss_type):
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        loss = mean_squared_error
    elif loss_type == 'mae':
        loss = mean_absolute_error
    elif loss_type == 'weighted_mse':
        loss = weighted_mse_loss
    elif loss_type == 'smooth_l1':
        loss = smooth_l1_loss
    elif loss_type == 'huber':
        loss = huber_loss
    else:
        raise ValueError('Unsupported loss type', loss_type)

    return loss
