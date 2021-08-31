#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import cv2
import os, sys, argparse
import numpy as np
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from hourglass.data import HG_OUTPUT_STRIDE
from hourglass.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import preprocess_image
from common.utils import get_classes, get_skeleton, render_skeleton

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_heatmap(heatmap, image, scale, class_names, skeleton_lines):
    start = time.time()
    # parse out predicted keypoint from heatmap
    keypoints = post_process_heatmap_simple(heatmap)

    # rescale keypoints back to origin image size
    keypoints_dict = dict()
    for i, keypoint in enumerate(keypoints):
        keypoints_dict[class_names[i]] = (keypoint[0] * scale[0] * HG_OUTPUT_STRIDE, keypoint[1] * scale[1] * HG_OUTPUT_STRIDE, keypoint[2])

    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    print('Keypoints detection result:')
    for keypoint in keypoints_dict.items():
        print(keypoint)

    # draw the keypoint skeleton on image
    image_array = np.array(image, dtype='uint8')
    image_array = render_skeleton(image_array, keypoints_dict, skeleton_lines)

    Image.fromarray(image_array).show()
    return


def validate_hourglass_model(model_path, image_file, class_names, skeleton_lines, model_image_size, loop_count):
    model = load_model(model_path, compile=False)

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_image_size[0], image_size[1] * 1.0 / model_image_size[1])

    # predict once first to bypass the model building time
    model.predict(image_data)

    start = time.time()
    for i in range(loop_count):
        prediction = model.predict(image_data)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    process_heatmap(heatmap, img, scale, class_names, skeleton_lines)
    return


def validate_hourglass_model_tflite(model_path, image_file, class_names, skeleton_lines, loop_count):
    interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    model_image_size = (height, width)

    image_data = preprocess_image(img, model_image_size)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_image_size[0], image_size[1] * 1.0 / model_image_size[1])

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    heatmap = prediction[-1][0]
    process_heatmap(heatmap, img, scale, class_names, skeleton_lines)
    return


def validate_hourglass_model_mnn(model_path, image_file, class_names, skeleton_lines, loop_count):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()

    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()
    if input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Tensorflow:
        batch, height, width, channel = input_shape
    elif input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        batch, channel, height, width = input_shape
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    model_image_size = (height, width)

    # prepare input image
    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_image_size[0], image_size[1] * 1.0 / model_image_size[1])

    # use a temp tensor to copy data
    tmp_input = MNN.Tensor(input_shape, input_tensor.getDataType(),\
                    image_data, input_tensor.getDimensionType())

    # predict once first to bypass the model building time
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    start = time.time()
    for i in range(loop_count):
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))


    # we only handle single output model
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    output_elementsize = reduce(mul, output_shape)

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                #np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    #tmp_output.printTensorData()

    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
    # our postprocess code based on TF channel last format, so if the output format
    # doesn't match, we need to transpose
    if output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        output_data = output_data.transpose((0,2,3,1))
    elif output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe_C4:
        raise ValueError('unsupported output tensor dimension type')

    heatmap = output_data[0]
    process_heatmap(heatmap, img, scale, class_names, skeleton_lines)


def validate_hourglass_model_onnx(model_path, image_file, class_names, skeleton_lines, loop_count):
    sess = onnxruntime.InferenceSession(model_path)

    input_tensors = []
    for i, input_tensor in enumerate(sess.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    batch, height, width, channel = input_tensors[0].shape
    model_image_size = (height, width)

    output_tensors = []
    for i, output_tensor in enumerate(sess.get_outputs()):
        output_tensors.append(output_tensor)
    # assume only 1 output tensor
    assert len(output_tensors) == 1, 'invalid output tensor number.'

    # prepare input image
    img = Image.open(image_file)
    image_data = preprocess_image(img, model_image_size)
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_image_size[0], image_size[1] * 1.0 / model_image_size[1])

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = sess.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = sess.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    process_heatmap(heatmap, img, scale, class_names, skeleton_lines)
    return


def validate_hourglass_model_pb(model_path, image_file, class_names, skeleton_lines, model_image_size, loop_count):
    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we need to hardcode the input/output tensor names here to get them from model
    output_tensor_name = 'graph/1_conv_1x1_parts/BiasAdd:0'

    # assume only 1 input tensor for image
    input_tensor_name = 'graph/image_input:0'

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    #image_shape = img.size
    image_size = img.size
    scale = (image_size[0] * 1.0 / model_image_size[0], image_size[1] * 1.0 / model_image_size[1])

    #load frozen pb graph
    def load_pb_graph(model_path):
        # We parse the graph_def file
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="graph",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    graph = load_pb_graph(model_path)

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in graph.get_operations():
        #print(op.name, op.values())

    image_input = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    # predict once first to bypass the model building time
    with tf.Session(graph=graph) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            image_input: image_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=graph) as sess:
                prediction = sess.run(output_tensor, feed_dict={
                    image_input: image_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    heatmap = prediction[0]
    process_heatmap(heatmap, img, scale, class_names, skeleton_lines)


def main():
    parser = argparse.ArgumentParser(description='validate Hourglass model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default=%(default)s', type=str, required=False, default='../../configs/mpii_classes.txt')
    parser.add_argument('--skeleton_path', help='path to keypoint skeleton definitions, default=%(default)s', type=str, required=False, default=None)
    parser.add_argument('--model_image_size', help='model image input size as <height>x<width>, default=%(default)s', type=str, default='256x256')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))


    # support of tflite model
    if args.model_path.endswith('.tflite'):
        validate_hourglass_model_tflite(args.model_path, args.image_file, class_names, skeleton_lines, args.loop_count)
    # support of MNN model
    elif args.model_path.endswith('.mnn'):
        validate_hourglass_model_mnn(args.model_path, args.image_file, class_names, skeleton_lines, args.loop_count)
    ## support of TF 1.x frozen pb model
    elif args.model_path.endswith('.pb'):
        validate_hourglass_model_pb(args.model_path, args.image_file, class_names, skeleton_lines, model_image_size, args.loop_count)
    # support of ONNX model
    elif args.model_path.endswith('.onnx'):
        validate_hourglass_model_onnx(args.model_path, args.image_file, class_names, skeleton_lines, args.loop_count)
    ## normal keras h5 model
    elif args.model_path.endswith('.h5'):
        validate_hourglass_model(args.model_path, args.image_file, class_names, skeleton_lines, model_image_size, args.loop_count)
    else:
        raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
