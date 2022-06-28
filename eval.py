#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate PCK for Hourglass model on validation dataset
"""
import os, argparse, json
import numpy as np
import operator
from operator import mul
from functools import reduce
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import MNN
import onnxruntime

from hourglass.data import hourglass_dataset
from hourglass.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import invert_transform_keypoints, revert_keypoints
from common.model_utils import get_normalize
from common.utils import get_classes, get_skeleton, render_skeleton, optimize_tf_gpu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


def check_pred_keypoints(pred_keypoint, gt_keypoint, threshold, normalize):
    # check if ground truth keypoint is valid
    if gt_keypoint[0] > 1 and gt_keypoint[1] > 1:
        # calculate normalized euclidean distance between pred and gt keypoints
        distance = np.linalg.norm(gt_keypoint[0:2] - pred_keypoint[0:2]) / normalize
        if distance < threshold:
            # succeed prediction
            return 1
        else:
            # fail prediction
            return 0
    else:
        # invalid gt keypoint
        return -1


def keypoint_accuracy(pred_keypoints, gt_keypoints, threshold, normalize):
    assert pred_keypoints.shape[0] == gt_keypoints.shape[0], 'keypoint number mismatch'

    result_list = []
    for i in range(gt_keypoints.shape[0]):
        # compare pred keypoint with gt keypoint to get result
        result = check_pred_keypoints(pred_keypoints[i, :], gt_keypoints[i, :], threshold, normalize)
        result_list.append(result)

    return result_list


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
      plt.barh(range(n_classes), sorted_values, color=plot_color)
      """
       Write number on side of bar
      """
      fig = plt.gcf() # gcf - get current figure
      axes = plt.gca()
      r = fig.canvas.get_renderer()
      for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
              str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
              adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15    # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def revert_pred_keypoints(keypoints, metainfo, model_input_shape, heatmap_shape):
    # invert transform keypoints based on center & scale
    center = metainfo['center']
    scale = metainfo['scale']
    image_shape = metainfo['image_shape']

    #######################################################################################################
    # 2 ways of keypoints invert transform, according to data preprocess solutions in hourglass/data.py

    # Option 1 (from origin repo):
    reverted_keypoints = invert_transform_keypoints(keypoints, center, scale, heatmap_shape, rotate_angle=0)

    # Option 2:
    #reverted_keypoints = revert_keypoints(keypoints, center, scale, image_shape, model_input_shape)

    return reverted_keypoints


def save_keypoints_detection(pred_keypoints, metainfo, class_names, skeleton_lines):
    result_dir=os.path.join('result','detection')
    os.makedirs(result_dir, exist_ok=True)

    image_name = metainfo['name']
    image = Image.open(image_name).convert('RGB')
    image_array = np.array(image, dtype='uint8')

    gt_keypoints = metainfo['pts']

    # form up gt keypoints & predict keypoints dict
    gt_keypoints_dict = {}
    pred_keypoints_dict = {}

    for i, keypoint in enumerate(gt_keypoints):
        gt_keypoints_dict[class_names[i]] = (keypoint[0], keypoint[1], 1.0)

    for i, keypoint in enumerate(pred_keypoints):
        pred_keypoints_dict[class_names[i]] = (keypoint[0], keypoint[1], keypoint[2])

    # render gt and predict keypoints skeleton on image
    image_array = render_skeleton(image_array, gt_keypoints_dict, skeleton_lines, colors=(255, 255, 255))
    image_array = render_skeleton(image_array, pred_keypoints_dict, skeleton_lines)

    image = Image.fromarray(image_array)
    # here we handle the RGBA image
    if(len(image.split()) == 4):
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    image.save(os.path.join(result_dir, image_name.split(os.path.sep)[-1]))
    return


def hourglass_predict_keras(model, image_data):
    prediction = model.predict(image_data)

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    return heatmap


def hourglass_predict_tflite(interpreter, image_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    #if input_details[0]['dtype'] == np.float32:
        #floating_model = True

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    model_input_shape = (height, width)

    image_data = image_data.astype('float32')
    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    heatmap = prediction[-1][0]
    return heatmap


def hourglass_predict_pb(model, image_data):
    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we need to hardcode the input/output tensor names here to get them from model
    output_tensor_name = 'graph/hg1_conv_1x1_predict/BiasAdd:0'

    # assume only 1 input tensor for image
    input_tensor_name = 'graph/image_input:0'

    # get input/output tensors
    image_input = model.get_tensor_by_name(input_tensor_name)
    output_tensor = model.get_tensor_by_name(output_tensor_name)

    with tf.Session(graph=model) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            image_input: image_data
        })
    heatmap = prediction[0]
    return heatmap


def hourglass_predict_onnx(model, image_data, class_names):
    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    # check if input layout is NHWC or NCHW
    if input_tensors[0].shape[1] == 3:
        batch, channel, height, width = input_tensors[0].shape  #NCHW
    else:
        batch, height, width, channel = input_tensors[0].shape  #NHWC

    output_tensors = []
    for i, output_tensor in enumerate(model.get_outputs()):
        output_tensors.append(output_tensor)
    # assume only 1 output tensor
    assert len(output_tensors) == 1, 'invalid output tensor number.'

    # check if output layout is NHWC or NCHW
    #if output_tensors[0].shape[1] == len(class_names):
        #print("NCHW output layout")
    #elif output_tensors[0].shape[-1] == len(class_names):
        #print("NHWC output layout")
    #else:
        #raise ValueError('invalid output layout or shape')

    if input_tensors[0].shape[1] == 3:
        # transpose image for NCHW layout
        image_data = image_data.transpose((0,3,1,2))

    feed = {input_tensors[0].name: image_data}
    prediction = model.run(None, feed)

    if output_tensors[0].shape[1] == len(class_names):
        # transpose predict mask for NCHW layout
        prediction = [p.transpose((0,2,3,1)) for p in prediction]

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    return heatmap


def hourglass_predict_mnn(interpreter, session, image_data, class_names):
    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)

    # get & resize input shape
    input_shape = list(input_tensor.getShape())
    if input_shape[0] == 0:
        input_shape[0] = 1
        interpreter.resizeTensor(input_tensor, tuple(input_shape))
        interpreter.resizeSession(session)

    # check if input layout is NHWC or NCHW
    if input_shape[1] == 3:
        batch, channel, height, width = input_shape  #NCHW
    elif input_shape[-1] == 3:
        batch, height, width, channel = input_shape  #NHWC
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    # create a temp tensor to copy data,
    # use TF NHWC layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    tmp_input_shape = (batch, height, width, channel)
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(image_data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    # we only handle single output model
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()

    # check if output layout is NHWC or NCHW
    #if output_shape[1] == len(class_names):
        #print("NCHW output layout")
    #elif output_shape[-1] == len(class_names):
        #print("NHWC output layout")
    #else:
        #raise ValueError('invalid output layout or shape')

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    output_elementsize = reduce(mul, output_shape)
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    #tmp_output.printTensorData()

    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
    # our postprocess code based on TF NHWC layout, so if the output format
    # doesn't match, we need to transpose
    if output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe and output_shape[1] == len(class_names): # double check if it's NCHW format
        output_data = output_data.transpose((0,2,3,1))
    elif output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe_C4:
        raise ValueError('unsupported output tensor dimension type')

    heatmap = output_data[0]
    return heatmap


def get_result_dict(pred_keypoints, metainfo):
    '''
     form up coco result dict with following format:
     {
      "image_id": int,
      "category_id": int,
      "keypoints": [x1,y1,v1,...,xk,yk,vk],
      "score": float
     }
    '''
    image_name = metainfo['name']
    image_id = int(os.path.basename(image_name).split('.')[0])

    result_dict = {}
    keypoints_list = []
    result_score = 0.0
    for i, keypoint in enumerate(pred_keypoints):
        keypoints_list.append(keypoint[0])
        keypoints_list.append(keypoint[1])
        keypoints_list.append(1) #visibility value. simply set vk=1
        result_score += keypoint[2]

    result_dict['image_id'] = image_id
    result_dict['category_id'] = 1  #person id
    result_dict['keypoints'] = keypoints_list
    result_dict['score'] = result_score/len(pred_keypoints)
    return result_dict


def eval_PCK(model, model_format, eval_dataset, class_names, model_input_shape, score_threshold, normalize, conf_threshold, save_result=False, skeleton_lines=None):
    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    succeed_dict = {class_name: 0 for class_name in class_names}
    fail_dict = {class_name: 0 for class_name in class_names}
    accuracy_dict = {class_name: 0. for class_name in class_names}

    # init output list for coco result json generation
    # coco keypoints result is a list of following format dict:
    # {
    #  "image_id": int,
    #  "category_id": int,
    #  "keypoints": [x1,y1,v1,...,xk,yk,vk],
    #  "score": float
    # }
    #
    output_list = []

    count = 0
    pbar = tqdm(total=eval_dataset.get_dataset_size(), desc='Eval model')
    for image_data, gt_heatmap, metainfo in eval_dataset:
        # fetch validation data from generator, which will crop out single person area, resize to input_shape and normalize image
        count += 1
        if count > eval_dataset.get_dataset_size():
            break

        # support of tflite model
        if model_format == 'TFLITE':
            heatmap = hourglass_predict_tflite(model, image_data)
        # support of MNN model
        elif model_format == 'MNN':
            heatmap = hourglass_predict_mnn(model, session, image_data, class_names)
        # support of TF 1.x frozen pb model
        elif model_format == 'PB':
            heatmap = hourglass_predict_pb(model, image_data)
        # support of ONNX model
        elif model_format == 'ONNX':
            heatmap = hourglass_predict_onnx(model, image_data, class_names)
        # normal keras h5 model
        elif model_format == 'H5':
            heatmap = hourglass_predict_keras(model, image_data)
        else:
            raise ValueError('invalid model format')

        heatmap_shape = heatmap.shape[0:2]

        # get predict keypoints from heatmap
        pred_keypoints = post_process_heatmap_simple(heatmap, conf_threshold)
        pred_keypoints = np.array(pred_keypoints)

        # get ground truth keypoints (transformed)
        metainfo = metainfo[0]
        gt_keypoints = metainfo['tpts']

        # calculate succeed & failed keypoints for prediction
        result_list = keypoint_accuracy(pred_keypoints, gt_keypoints, score_threshold, normalize)

        for i, class_name in enumerate(class_names):
            if result_list[i] == 0:
                fail_dict[class_name] = fail_dict[class_name] + 1
            elif result_list[i] == 1:
                succeed_dict[class_name] = succeed_dict[class_name] + 1

        # revert predict keypoints back to origin image shape
        reverted_pred_keypoints = revert_pred_keypoints(pred_keypoints, metainfo, model_input_shape, heatmap_shape)

        # get coco result dict with predict keypoints and image info
        result_dict = get_result_dict(reverted_pred_keypoints, metainfo)
        # add result dict to output list
        output_list.append(result_dict)

        if save_result:
            # render keypoints skeleton on image and save result
            save_keypoints_detection(reverted_pred_keypoints, metainfo, class_names, skeleton_lines)
        pbar.update(1)
    pbar.close()

    # save to coco result json
    os.makedirs('result', exist_ok=True)
    json_fp = open(os.path.join('result','keypoints_result.json'), 'w')
    json_str = json.dumps(output_list)
    json_fp.write(json_str)
    json_fp.close()

    # calculate accuracy for each class
    for i, class_name in enumerate(class_names):
        accuracy_dict[class_name] = succeed_dict[class_name] * 1.0 / (succeed_dict[class_name] + fail_dict[class_name])

    #get PCK accuracy from succeed & failed keypoints
    total_succeed = np.sum(list(succeed_dict.values()))
    total_fail = np.sum(list(fail_dict.values()))
    total_accuracy = total_succeed * 1.0 / (total_fail + total_succeed)

    if save_result:
        '''
         Draw PCK plot
        '''
        window_title = "PCK evaluation"
        plot_title = "PCK@{0} score = {1:.2f}%".format(score_threshold, total_accuracy)
        x_label = "Accuracy"
        output_path = os.path.join('result','PCK.png')
        draw_plot_func(accuracy_dict, len(accuracy_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    return total_accuracy, accuracy_dict


#load TF 1.x frozen pb graph
def load_graph(model_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

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


def load_eval_model(model_path):
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)
        model_format = 'MNN'

    # support of TF 1.x frozen pb model
    elif model_path.endswith('.pb'):
        model = load_graph(model_path)
        model_format = 'PB'

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)
        model_format = 'ONNX'

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        model = load_model(model_path, compile=False)
        model_format = 'H5'
        K.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model, model_format


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate Hourglass model (h5/pb/onnx/tflite/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to class definitions, default=%(default)s', default='configs/mpii_classes.txt')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='dataset path containing images and annotation file')

    parser.add_argument(
        '--score_threshold', type=float,
        help='score threshold for PCK evaluation, default=%(default)s', default=0.5)

    #parser.add_argument(
        #'--normalize', type=float,
        #help='normalized coefficient of keypoint distance for PCK evaluation , default=6.4', default=6.4)

    parser.add_argument(
        '--conf_threshold', type=float,
        help='confidence threshold for filtering keypoint in postprocess, default=%(default)s', default=1e-6)

    parser.add_argument(
        '--model_input_shape', type=str,
        help='model image input shape as <height>x<width>, default=%(default)s', default='256x256')

    parser.add_argument(
        '--save_result', default=False, action="store_true",
        help='Save the detection result image in result/detection dir')

    parser.add_argument(
        '--skeleton_path', type=str, required=False,
        help='path to keypoint skeleton definitions, default None', default=None)

    args = parser.parse_args()

    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))
    normalize = get_normalize(model_input_shape)

    # load trained model for eval
    model, model_format = load_eval_model(args.model_path)

    # prepare eval dataset
    eval_dataset = hourglass_dataset(args.dataset_path, batch_size=1, class_names=class_names,
                              input_shape=model_input_shape, num_hgstack=1, is_train=False, with_meta=True)


    total_accuracy, accuracy_dict = eval_PCK(model, model_format, eval_dataset, class_names, model_input_shape, args.score_threshold, normalize, args.conf_threshold, args.save_result, skeleton_lines)

    print('\nPCK evaluation')
    for (class_name, accuracy) in accuracy_dict.items():
        print('%s: %f' % (class_name, accuracy))
    print('total acc: %f' % (total_accuracy))


if __name__ == '__main__':
    main()
