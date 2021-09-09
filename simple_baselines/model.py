#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train CNN classifier on images split into directories.
"""
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50

import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.backbones.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from common.backbones.peleenet import PeleeNet
from common.backbones.ghostnet import GhostNet
from simple_baselines.layers import Deconv_block, Upsample_block, Upsample_lite_block


def get_base_model(model_type, model_input_shape, weights='imagenet'):
    #prepare input tensor
    if model_input_shape:
        input_tensor = Input(shape=model_input_shape+(3,), name='image_input')
    else:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    if model_type.startswith('mobilenetv1'):
        model = MobileNet(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False, alpha=1.0)
    elif model_type.startswith('mobilenetv2'):
        model = MobileNetV2(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False, alpha=1.0)
    elif model_type.startswith('mobilenetv3large'):
        model = MobileNetV3Large(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False, alpha=1.0)
    elif model_type.startswith('mobilenetv3small'):
        model = MobileNetV3Small(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False, alpha=1.0)
    elif model_type.startswith('peleenet'):
        model = PeleeNet(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False)
    elif model_type.startswith('ghostnet'):
        model = GhostNet(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False)
    elif model_type.startswith('resnet50'):
        model = ResNet50(input_tensor=input_tensor, weights=weights, pooling=None, include_top=False)
    else:
        raise ValueError('Unsupported model type', model_type)

    return model


def get_simple_baselines_model(model_type, num_classes, model_input_shape=None, weights_path=None):
    # create the base pre-trained model
    base_model = get_base_model(model_type, model_input_shape)
    backbone_len = len(base_model.layers)

    x = base_model.output

    # Simple Baselines Deconv block config
    num_layers = 3
    deconv_channels = [256, 256, 256]
    deconv_kernels = [4, 4, 4]
    use_bias=True

    # Deconv/Upsample block
    if model_type.endswith('_deconv'):
        x = Deconv_block(x, num_layers, deconv_channels, deconv_kernels, use_bias=use_bias)
    elif model_type.endswith('_upsample'):
        x = Upsample_block(x, num_layers, deconv_channels, deconv_kernels, use_bias=use_bias)
    elif model_type.endswith('_upsample_lite'):
        x = Upsample_lite_block(x, num_layers, deconv_channels, deconv_kernels, use_bias=use_bias)
    else:
        raise ValueError('Unsupported model type', model_type)

    # final heatmap prediction, use 'linear' activation.
    prediction = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name='heatmap_predict')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model, backbone_len



if __name__ == '__main__':
    model, _ = get_simple_baselines_model('mobilenetv2_deconv', 21, (256, 256))
    model.summary()
    #model.save('check.h5')

