#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stacked Hourglass model blocks definition in Keras."""
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Add, UpSampling2D
import tensorflow.keras.backend as K


def bottleneck_block(x, num_out_channels, block_name):
    """
    normal hourglass bottleneck block,
    using standard conv
    """
    # skip layer
    if K.int_shape(x)[-1] == num_out_channels:
        _skip = x
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                       name=block_name + '_skip')(x)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv2D(num_out_channels//2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_1')(x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels//2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_3')(_x)
    _x = BatchNormalization()(_x)

    # merge residual and skip branch
    _x = Add(name=block_name + '_add')([_skip, _x])

    return _x


def bottleneck_mobile(x, num_out_channels, block_name):
    """
    lightweight hourglass bottleneck block,
    using separable conv
    """
    # skip layer
    if K.int_shape(x)[-1] == num_out_channels:
        _skip = x
    else:
        _skip = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                                name=block_name + '_skip')(x)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_1')(x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                         name=block_name + '_conv_3x3_2')(_x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_3')(_x)
    _x = BatchNormalization()(_x)

    # merge residual and skip branch
    _x = Add(name=block_name + '_add')([_skip, _x])

    return _x


def create_front_module(input_tensor, num_channels, bottleneck):
    """
    front module block, input to 1/4 resolution:
      input_tensor: 256 x 256 x 3
      front_residual_3 output: 64 x 64 x num_channels

    using following blocks:
      * 1 7x7 conv + maxpooling
      * 3 residual block
    """
    _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_1')(input_tensor)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels//2, 'front_residual_1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_channels//2, 'front_residual_2')
    _x = bottleneck(_x, num_channels, 'front_residual_3')

    return _x


def hourglass_module(x, num_classes, num_channels, bottleneck, hg_id):
    """
    single hourglass module block
      1. downsample feature map to 1/8 size
      2. upsample back to origin
      3. create 2 heads for next stage and loss/prediction
    """
    # create downsample features: f1, f2, f4 and f8
    downsample_features = create_downsample_blocks(x, bottleneck, hg_id, num_channels)

    # create upsample features, and connect with downsample features
    upsample_feature = create_upsample_blocks(downsample_features, bottleneck, hg_id, num_channels)

    # add 1x1 conv with two heads, head_next_stage is for next stage hourglass,
    # head_predict is for intermediate supervision (loss) or final prediction
    head_next_stage, head_predict = create_heads(x, upsample_feature, num_classes, hg_id, num_channels)

    return head_next_stage, head_predict


def create_downsample_blocks(x, bottleneck, hg_id, num_channels):
    """
    create downsample blocks and 4 different scale features for hourglass module
      input image: 256 x 256 x 3
      x: 64 x 64 x num_channels

      downsample_f1 feature: 64 x 64 x num_channels
      downsample_f2 feature: 32 x 32 x num_channels
      downsample_f4 feature: 16 x 16 x num_channels
      downsample_f8 feature: 8 x 8 x num_channels
    """
    hg_name = 'hg' + str(hg_id)

    downsample_f1 = bottleneck(x, num_channels, hg_name + '_downsample_1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f1)

    downsample_f2 = bottleneck(_x, num_channels, hg_name + '_downsample_2')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f2)

    downsample_f4 = bottleneck(_x, num_channels, hg_name + '_downsample_4')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downsample_f4)

    downsample_f8 = bottleneck(_x, num_channels, hg_name + '_downsample_8')

    return (downsample_f1, downsample_f2, downsample_f4, downsample_f8)


def connect_downsample_upsample(downsample_feature, upsample_feature, bottleneck, num_channels, name):
    """
    merge downsample feature and low scale upsample feature to
    create high scale upsample feature:

    1. apply 1 bottleneck block for downsample feature
    2. upsampling low scale upsample feature
    3. merge downsample && upsample features
    4. apply 1 bottleneck block for merged feature
    """
    downsample_x = bottleneck(downsample_feature, num_channels, name + '_short')

    upsample_x = UpSampling2D()(upsample_feature)

    # merge downsample && upsample features
    x = Add()([downsample_x, upsample_x])
    # apply bottleneck block to merged feature
    out = bottleneck(x, num_channels, name + '_merged')

    return out


def bottom_block(downsample_f8, bottleneck, hg_id, num_channels):
    """
    blocks in lowest resolution(8x8 for 256x256 input) to
    create upsample_f8 from downsample_f8:
      1. 1 bottleneck block for shortcut
      2. 3 bottleneck blocks for main branch
      3. add shortcut and main
    """
    downsample_f8_short = bottleneck(downsample_f8, num_channels, str(hg_id) + "_downsample_f8_short")

    _x = bottleneck(downsample_f8, num_channels, str(hg_id) + "_downsample_f8_1")
    _x = bottleneck(_x, num_channels, str(hg_id) + "_downsample_f8_2")
    _x = bottleneck(_x, num_channels, str(hg_id) + "_downsample_f8_3")

    upsample_f8 = Add()([_x, downsample_f8_short])

    return upsample_f8


def create_upsample_blocks(downsample_features, bottleneck, hg_id, num_channels):
    """
    do upsample and merge 4 different scale downsample features to get final upsample feature map

      downsample_f1 feature: 64 x 64 x num_channels
      downsample_f2 feature: 32 x 32 x num_channels
      downsample_f4 feature: 16 x 16 x num_channels
      downsample_f8 feature: 8 x 8 x num_channels

      upsample_f8 feature: 8 x 8 x num_channels
      upsample_f4 feature: 16 x 16 x num_channels
      upsample_f2 feature: 32 x 32 x num_channels
      upsample_f1 feature: 64 x 64 x num_channels
    """
    downsample_f1, downsample_f2, downsample_f4, downsample_f8 = downsample_features

    upsample_f8 = bottom_block(downsample_f8, bottleneck, hg_id, num_channels)
    upsample_f4 = connect_downsample_upsample(downsample_f4, upsample_f8, bottleneck, num_channels, 'hg'+str(hg_id)+'_upsample_f4')
    upsample_f2 = connect_downsample_upsample(downsample_f2, upsample_f4, bottleneck, num_channels, 'hg'+str(hg_id)+'_upsample_f2')
    upsample_f1 = connect_downsample_upsample(downsample_f1, upsample_f2, bottleneck, num_channels, 'hg'+str(hg_id)+'_upsample_f1')

    return upsample_f1


def create_heads(x, upsample_feature, num_classes, hg_id, num_channels):
    """
    create two output heads for hourglass module
      * one head for next stage: head_next_stage
      * one head for intermediate supervision (loss) or final prediction: head_predict
    """
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hg_id) + '_conv_1x1_1')(upsample_feature)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_predict = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hg_id) + '_conv_1x1_predict')(head)

    # use linear activation
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hg_id) + '_conv_1x1_2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hg_id) + '_conv_1x1_3')(head_predict)

    # merge heads for next stage
    head_next_stage = Add()([head, head_m, x])

    return head_next_stage, head_predict

