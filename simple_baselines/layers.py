#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, SeparableConv2D, BatchNormalization, ReLU
#from tensorflow.keras import backend as K


BN_MOMENTUM = 0.1

def get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 'same'
        output_padding = None
    elif deconv_kernel == 3:
        padding = 'same'
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 'valid'
        output_padding = None

    return deconv_kernel, padding, output_padding


def Deconv_block(x, num_layers, deconv_channels, deconv_kernels, use_bias):
    assert num_layers == len(deconv_channels), \
        'ERROR: deconv layer number is different with len(deconv_channels)'
    assert num_layers == len(deconv_kernels), \
        'ERROR: deconv layer number is different with len(deconv_kernels)'

    for i in range(num_layers):
        kernel, padding, output_padding = get_deconv_cfg(deconv_kernels[i], i)

        x = Conv2DTranspose(deconv_channels[i],
                            kernel,
                            strides=(2, 2),
                            padding=padding,
                            output_padding=output_padding,
                            use_bias=use_bias)(x)
        x = BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = ReLU()(x)

    return x


def Upsample_block(x, num_layers, deconv_channels, deconv_kernels, use_bias):
    assert num_layers == len(deconv_channels), \
        'ERROR: deconv layer number is different with len(deconv_channels)'
    assert num_layers == len(deconv_kernels), \
        'ERROR: deconv layer number is different with len(deconv_kernels)'

    for i in range(num_layers):
        # conv kernel based on deconv config
        kernel = 3 if deconv_kernels[i] >= 3 else 1

        # use UpSample+Conv2D to replace Deconv
        x = UpSampling2D(2)(x)
        x = Conv2D(deconv_channels[i],
                   kernel,
                   strides=1,
                   padding='same',
                   use_bias=use_bias)(x)
        x = BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = ReLU()(x)

    return x


def Upsample_lite_block(x, num_layers, deconv_channels, deconv_kernels, use_bias):
    assert num_layers == len(deconv_channels), \
        'ERROR: deconv layer number is different with len(deconv_channels)'
    assert num_layers == len(deconv_kernels), \
        'ERROR: deconv layer number is different with len(deconv_kernels)'

    for i in range(num_layers):
        # conv kernel based on deconv config
        kernel = 3 if deconv_kernels[i] >= 3 else 1

        # use UpSample+SeparableConv2D to replace Deconv
        x = UpSampling2D(2)(x)
        x = SeparableConv2D(deconv_channels[i],
                            deconv_kernels[i]-1,
                            strides=1,
                            padding='same',
                            use_bias=use_bias)(x)
        x = BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = ReLU()(x)

    return x

