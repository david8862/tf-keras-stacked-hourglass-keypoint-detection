#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K


def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))
