#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert Hourglass keras model to an integer quantized tflite model
using Post-Training Integer Quantization Toolkit released in tensorflow
2.0.0 build
"""
import os, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from hourglass.data import hourglass_dataset
from common.utils import get_classes
#from common.utils import get_custom_objects

#tf.enable_eager_execution()


def post_train_quant_convert(keras_model_file, dataset_path, class_names, sample_num, output_file):
    #custom_object_dict = get_custom_objects(custom_objects_string)

    model = load_model(keras_model_file, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # get input shape, assume only 1 input
    input_shape = tuple(model.input.shape.as_list()[1:3])

    # prepare representative dataset
    represent_data = hourglass_dataset(dataset_path, batch_size=1, class_names=class_names,
                          input_shape=input_shape, num_hgstack=1, is_train=False, with_meta=False)

    def data_generator():
        i = 0
        #for num in range(sample_num):
        for image, gt_heatmap in represent_data:
            i = i+1
            if i >= sample_num:
                break
            image = np.array(image, dtype=np.float32)
            yield [image]


    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.representative_dataset = tf.lite.RepresentativeDataset(data_generator)

    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_model)



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='TF 2.x post training integer quantization converter for Hourglass keypoint detection model')

    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--dataset_path', required=True, type=str, help='dataset path containing images and annotation file')
    parser.add_argument('--classes_path', required=True, type=str, help='path to keypoint class definition file')
    parser.add_argument('--sample_num', type=int, help='image sample number to feed the converter, default=%(default)s', default=100)
    parser.add_argument('--output_file', required=True, type=str, help='output tflite model file')

    args = parser.parse_args()

    # param parse
    class_names = get_classes(args.classes_path)

    post_train_quant_convert(args.keras_model_file, args.dataset_path, class_names, args.sample_num, args.output_file)
    return



if __name__ == '__main__':
    main()

