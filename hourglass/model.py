#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from hourglass.blocks import create_front_module, hourglass_module, bottleneck_block, bottleneck_mobile


def get_hourglass_model(num_classes, num_stacks, num_channels, input_size, mobile):
    # whether to use depthwise conv use choose model type
    if mobile:
        bottleneck = bottleneck_mobile
    else:
        bottleneck = bottleneck_block

    input_tensor = Input(shape=(input_size[0], input_size[1], 3), name='image_input')

    # front module, input to 1/4 resolution
    front_features = create_front_module(input_tensor, num_channels, bottleneck)

    # form up hourglass stacks and get head of
    # each module for intermediate supervision
    head_next_stage = front_features
    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
        outputs.append(head_to_loss)

    model = Model(inputs=input_tensor, outputs=outputs)

    return model

