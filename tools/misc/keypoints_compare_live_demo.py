#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
from timeit import default_timer as timer
from threading import Thread
import json
import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

from keypoints_compare import keypoints_compare, person_action_check, SIMILARITY_SCORE_THRESHOLD_HIGH, SIMILARITY_SCORE_THRESHOLD_LOW

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from annotate_keypoints_multiperson import predict_hourglass_model, predict_hourglass_model_tflite, predict_hourglass_model_mnn, predict_hourglass_model_onnx, predict_hourglass_model_pb, load_val_model, append_info_to_json
from hourglass.postprocess import HG_OUTPUT_STRIDE, post_process_heatmap, post_process_heatmap_simple
from hourglass.utils import preprocess_image, get_classes, get_skeleton, render_skeleton, optimize_tf_gpu
from detector import detect_person, get_anchors, get_square_box

DET_ANCHORS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'detector', 'yolo3_anchors.txt')
DET_CLASSES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'detector', 'coco_classes.txt')
DET_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'detector', 'yolo3_mobilenet_lite_320_coco.h5')
DET_MODEL_INPUT_SHAPE = (320, 320)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


def parse_json_data(json_data, class_names):
    """
    Parse person bbox & raw keypoints from labelme format json data

    # Arguments
        json_data: labelme json format keypoints info,
            python dict with labelme struct, see related doc
        class_names: keypoint class name list,
            list of string with keypoints name

    # Returns
        keypoints: raw keypoints coordinate,
            dict of points with shape (num_keypoints, 2),
    """
    # init empty bbox list & keypoints dict
    bbox = []
    keypoints_dict = {}

    # parse person bbox
    for shape in json_data['shapes']:
        if shape['label'] == 'person':
            bbox = list(map(int, [shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]]))

    # parse person keypoints
    class_count = 0
    for i in range(len(class_names)):
        for shape in json_data['shapes']:
            if shape['label'] == class_names[i]:
                # fill keypoints with raw coordinate
                keypoints_dict[class_names[i]] = (shape['points'][0][0], shape['points'][0][1], 1.0)
                class_count += 1
        if class_count != i+1:
            # fill 0 if no valid coordinate
            keypoints_dict[class_names[i]] = (0.0, 0.0, 1.0)
            class_count += 1

    assert class_count == len(class_names), 'keypoint number mismatch'
    return bbox, keypoints_dict


# global value to temp record 'critical' coach action keypoints dict
COACH_DICT = dict()

def coach_thread_func(args):
    global COACH_DICT
    # param parse
    class_names = get_classes(args.classes_path)
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    # get image file list and annotation json file list
    image_files = sorted(glob.glob(os.path.join(args.annotation_path, '*.jpg')))
    json_files = sorted(glob.glob(os.path.join(args.annotation_path, '*.json')))
    assert len(image_files) == len(json_files), 'annotation file mismatch.'

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    i = 0
    while (True):
        # loop back to 1st frame
        if i >= len(image_files):
            i = 0

        image_file = image_files[i]
        json_file = json_files[i]
        i += 1

        # load & parse json data
        with open(json_file) as f:
            annotate_data = json.load(f)

        # load coach image
        image = cv2.imread(image_file)

        # calculate & show FPS
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        # check if annotate json data has 'critical' flag, and show it on screen
        if len(annotate_data['flags']) != 0 and annotate_data['flags']['critical'] == True:
            cv2.putText(image, text='critical', org=(image.shape[1]-100, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)

        bbox, keypoints_dict = parse_json_data(annotate_data, class_names)
        # draw bbox rectangle on image
        if len(bbox) > 0:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1, cv2.LINE_AA)
        if len(keypoints_dict) > 0:
            # render keypoints skeleton on image
            image = render_skeleton(image, keypoints_dict, skeleton_lines)

        #cv2.namedWindow("Coach", cv2.WINDOW_NORMAL)
        cv2.imshow("Coach", image)

        keycode = cv2.waitKey(15) # delay 30 ms
        if keycode == ord('q'):
            print('exit')
            exit()

        # for 'critical' coach frame, record it for user thread check, and pending for 500ms
        if len(annotate_data['flags']) != 0 and annotate_data['flags']['critical'] == True:
            COACH_DICT = annotate_data
            time.sleep(0.5)
        elif len(COACH_DICT) > 0:
            # clear 'critical' info in next frame
            COACH_DICT = dict()


def user_thread_func(args):
    global COACH_DICT
    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    # prepare detection model and configs
    det_anchors = get_anchors(DET_ANCHORS_PATH)
    det_class_names = get_classes(DET_CLASSES_PATH)
    det_model = load_model(DET_MODEL_PATH, compile=False)
    det_model_input_shape = DET_MODEL_INPUT_SHAPE

    # prepare hourglass model
    model = load_val_model(args.model_path)
    if args.model_path.endswith('.mnn'):
        #MNN inference engine need create session
        session_config = \
        {
          'backend': 'CPU',  #'CPU'/'OPENCL'/'OPENGL'/'VULKAN'/'METAL'/'TRT'/'CUDA'/'HIAI'
          'precision': 'high',  #'normal'/'low'/'high'/'lowBF'
          'numThread': 2
        }
        session = model.createSession(session_config)

    # capture user video from web camera
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam")

    best_score = 0
    best_dict = dict()
    best_feedback = ''
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        # read video frame
        ret, frame = vid.read()
        if ret != True:
            break

        image = Image.fromarray(frame)
        image_array = np.array(image, dtype='uint8')
        width, height = image.size

        # labelme json data struct
        user_dict = {"version": "4.6.0",
                     "flags": {},
                     "shapes": [],
                     "imagePath": None,
                     "imageData": None,
                     "imageHeight": height,
                     "imageWidth": width
                    }

        # detect person bbox from origin image
        person_boxes, person_scores = detect_person(image, det_model, det_anchors, det_class_names, det_model_input_shape)

        # here we bypass keypoint detection if more than 1 person bbox
        if len(person_boxes) > 1:
            cv2.namedWindow("User", cv2.WINDOW_NORMAL)
            cv2.imshow("User", np.asarray(image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        keypoints_dict = dict()
        for i, (box, score) in enumerate(zip(person_boxes, person_scores)):
            raw_xmin, raw_ymin, raw_xmax, raw_ymax = map(int, box)

            # expand person bbox to square
            xmin, ymin, xmax, ymax = get_square_box(box, image.size)

            # crop person image area as keypoint model input
            person_image = Image.fromarray(image_array[ymin:ymax, xmin:xmax])
            person_array = np.array(person_image, dtype='uint8')

            # NOTE: image_size and scale in (w,h) format, but
            #       model_input_shape in (h,w) format
            image_size = person_image.size
            scale = (image_size[0] * 1.0 / model_input_shape[1], image_size[1] * 1.0 / model_input_shape[0])

            # support of tflite model
            if args.model_path.endswith('.tflite'):
                keypoints = predict_hourglass_model_tflite(model, person_image, args.conf_threshold, args.loop_count)
            # support of MNN model
            elif args.model_path.endswith('.mnn'):
                keypoints = predict_hourglass_model_mnn(model, session, person_image, args.conf_threshold, args.loop_count)
            ## support of TF 1.x frozen pb model
            elif args.model_path.endswith('.pb'):
                keypoints = predict_hourglass_model_pb(model, person_image, model_input_shape, args.conf_threshold, args.loop_count)
            # support of ONNX model
            elif args.model_path.endswith('.onnx'):
                keypoints = predict_hourglass_model_onnx(model, person_image, args.conf_threshold, args.loop_count)
            ## normal keras h5 model
            elif args.model_path.endswith('.h5'):
                keypoints = predict_hourglass_model(model, person_image, model_input_shape, args.conf_threshold, args.loop_count)
            else:
                raise ValueError('invalid model file')

            # rescale keypoints back to origin image shape
            keypoints_dict = dict()
            for j, keypoint in enumerate(keypoints):
                keypoints_dict[class_names[j]] = (keypoint[0] * scale[0] * HG_OUTPUT_STRIDE + xmin, keypoint[1] * scale[1] * HG_OUTPUT_STRIDE + ymin, keypoint[2])

            # form up labelme format json dict
            bbox = [raw_xmin, raw_ymin, raw_xmax, raw_ymax]
            user_dict = append_info_to_json(user_dict, bbox, keypoints_dict, i)

        image = np.asarray(image)
        if len(COACH_DICT) > 0:
            # compare critical coach keypoints and user keypoints
            score, distance_dict = keypoints_compare(COACH_DICT, user_dict, class_names, normalize_shape=(256, 256))

            # judge pose similarity with score first
            if score > SIMILARITY_SCORE_THRESHOLD_HIGH:
                feedback_string = 'perfect action\n'
            elif score > SIMILARITY_SCORE_THRESHOLD_LOW:
                # check person action with keypoints distance, and
                # show related feedback UI message
                feedback_string = person_action_check(distance_dict, COACH_DICT)
            else:
                feedback_string = 'action mismatch\n'

            # record best user keypoints during 'critical' time
            if score > best_score:
                best_score = score
                best_dict = distance_dict
                best_feedback = feedback_string
        else:
            # show best compare result
            if best_score > 0:
                print('best human pose similarity score:', best_score)
                print('\nbest keypoints distance:')
                for distance in best_dict.items():
                    print(distance)

                print('\nAction feedback:')
                print(best_feedback)

                # clear best record for checking next 'critical' frame
                best_score = 0
                best_dict = dict()
                best_feedback = ''

        # calculate & show FPS
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        # draw bbox rectangle on image
        cv2.rectangle(image, (raw_xmin, raw_ymin), (raw_xmax, raw_ymax), (255, 0, 0), 1, cv2.LINE_AA)
        if len(keypoints_dict) > 0:
            # render keypoints skeleton on user image
            image = render_skeleton(image, keypoints_dict, skeleton_lines)

        cv2.namedWindow("User", cv2.WINDOW_NORMAL)
        cv2.imshow("User", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Live demo for person keypoints compare with web camera')
    parser.add_argument('--model_path', help='keypoint model file', type=str, required=True)
    parser.add_argument('--classes_path', help='path to keypoint class definitions, default=%(default)s', type=str, required=False, default='../../configs/mpii_classes.txt')

    parser.add_argument('--annotation_path', help='path with annotation images and labelme jsons', type=str, required=True)
    parser.add_argument('--skeleton_path', help='path to keypoint skeleton definitions, default=%(default)s', type=str, required=False, default=None)
    parser.add_argument('--model_input_shape', help='keypoint model input shape as <height>x<width>, default=%(default)s', type=str, default='256x256')
    parser.add_argument('--conf_threshold', help='confidence threshold for filtering keypoint in postprocess, default=%(default)s', type=float, default=0.01)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # thread to show & record coach video
    coach_thread = Thread(target=coach_thread_func, args=(args,))
    coach_thread.start()

    # thread to capture & check user action from web camera
    user_thread = Thread(target=user_thread_func, args=(args,))
    user_thread.start()


if __name__ == '__main__':
    main()
