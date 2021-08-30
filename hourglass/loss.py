#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error#, huber
import tensorflow.keras.backend as K


def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))


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
    elif loss_type == 'smooth_l1':
        loss = smooth_l1_loss
    elif loss_type == 'huber':
        loss = huber_loss
    else:
        raise ValueError('Unsupported loss type', loss_type)

    return loss
