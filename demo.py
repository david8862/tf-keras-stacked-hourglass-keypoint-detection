#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import cv2
from PIL import Image
import numpy as np
import time
from timeit import default_timer as timer
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

from hourglass.model import get_hourglass_model
from hourglass.data import HG_OUTPUT_STRIDE
from hourglass.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import preprocess_image
from common.utils import get_classes, get_skeleton, render_skeleton, optimize_tf_gpu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

default_config = {
        "num_stacks": 2,
        "mobile" : False,
        "tiny" : False,
        "model_image_size": (256, 256),
        "num_channels": 256,
        "conf_threshold": 0.1,
        "classes_path": os.path.join('configs', 'mpii_classes.txt'),
        "skeleton_path": None,
        "weights_path": os.path.join('weights', 'hourglass_mobile.h5'),
        "gpu_num" : 1,
    }


class Hourglass(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(Hourglass, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        if self.skeleton_path:
            self.skeleton_lines = get_skeleton(self.skeleton_path)
        else:
            self.skeleton_lines = None
        self.class_names = get_classes(self.classes_path)
        self.hourglass_model = self._generate_model()
        K.set_learning_phase(0)

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_classes = len(self.class_names)

        # update param for tiny model
        if self.tiny is True:
            self.num_channels = 128

        # construct model and load weights.
        hourglass_model = get_hourglass_model(num_classes, self.num_stacks, self.num_channels, input_size=self.model_image_size, mobile=self.mobile)
        hourglass_model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        hourglass_model.summary()
        return hourglass_model


    def detect_image(self, image):
        image_data = preprocess_image(image, self.model_image_size)

        image_size = image.size
        scale = (image_size[0] * 1.0 / self.model_image_size[0], image_size[1] * 1.0 / self.model_image_size[1])

        start = time.time()
        keypoints = self.predict(image_data)
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        # rescale keypoints back to origin image size
        keypoints_dict = dict()
        for i, keypoint in enumerate(keypoints):
            keypoints_dict[self.class_names[i]] = (keypoint[0] * scale[0] * HG_OUTPUT_STRIDE, keypoint[1] * scale[1] * HG_OUTPUT_STRIDE, keypoint[2])

        # draw the keypoint skeleton on image
        image_array = np.array(image, dtype='uint8')
        image_array = render_skeleton(image_array, keypoints_dict, self.skeleton_lines, self.conf_threshold)

        return Image.fromarray(image_array)

    def predict(self, image_data):
        # get final predict heatmap
        prediction = self.hourglass_model.predict(image_data)
        if isinstance(prediction, list):
            prediction = prediction[-1]
        heatmap = prediction[0]

        # parse out predicted keypoint from heatmap
        keypoints = post_process_heatmap_simple(heatmap)

        return keypoints

    def dump_model_file(self, output_model_file):
        # Dump out the final heatmap output model as inference model,
        # since we don't need the intermediate heatmap in inference stage
        model = Model(inputs=self.hourglass_model.input, outputs=self.hourglass_model.outputs[-1])
        model.save(output_model_file)


def detect_video(hourglass, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = hourglass.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()



def detect_img(hourglass):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = hourglass.detect_image(image)
            r_image.show()



if __name__ == "__main__":
    # class Hourglass defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out Hourglass h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--num_stacks', type=int,
        help='num of stacks, default ' + str(Hourglass.get_defaults("num_stacks"))
    )
    parser.add_argument(
        '--mobile', default=False, action="store_true",
        help='use depthwise conv in hourglass, default ' + str(Hourglass.get_defaults("mobile"))
    )
    parser.add_argument(
        '--tiny', default=False, action="store_true",
        help='tiny network for speed, feature channel=128, default ' + str(Hourglass.get_defaults("tiny"))
    )
    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <num>x<num>, default ' +
        str(Hourglass.get_defaults("model_image_size")[0])+'x'+str(Hourglass.get_defaults("model_image_size")[1]),
        default=str(Hourglass.get_defaults("model_image_size")[0])+'x'+str(Hourglass.get_defaults("model_image_size")[1])
    )
    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + Hourglass.get_defaults("weights_path")
    )
    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to keypoint class definitions, default ' + Hourglass.get_defaults("classes_path")
    )
    parser.add_argument(
        '--skeleton_path', type=str, required=False,
        help='path to keypoint skeleton definitions, default ' + str(Hourglass.get_defaults("skeleton_path"))
    )
    parser.add_argument(
        '--conf_threshold', type=float,
        help='confidence threshold, default ' + str(Hourglass.get_defaults("conf_threshold"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    args = parser.parse_args()
    # param parse
    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        args.model_image_size = (int(height), int(width))

    # get wrapped inference object
    hourglass = Hourglass(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        hourglass.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in args:
            print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
        detect_img(hourglass)
    elif "input" in args:
        detect_video(hourglass, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

