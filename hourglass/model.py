#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""create Stacked Hourglass model for train."""
import os, sys
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from hourglass.blocks import create_front_module, hourglass_module, bottleneck_block, bottleneck_mobile


def get_hourglass_model(num_classes, num_stacks, num_channels, model_input_shape=None, mobile=False):
    # whether to use depthwise conv use choose model type
    if mobile:
        bottleneck = bottleneck_mobile
    else:
        bottleneck = bottleneck_block

    # prepare input tensor
    if model_input_shape:
        input_tensor = Input(shape=(model_input_shape[0], model_input_shape[1], 3), name='image_input')
    else:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    # front module, input to 1/4 resolution
    front_features = create_front_module(input_tensor, num_channels, bottleneck)

    # form up hourglass stacks and get head of
    # each module for intermediate supervision
    head_next_stage = front_features
    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
        outputs.append(head_to_loss)

    # create multi output model for intermediate supervision training process
    model = Model(inputs=input_tensor, outputs=outputs)

    return model

