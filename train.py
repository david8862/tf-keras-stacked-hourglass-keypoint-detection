#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN

from hourglass.model import get_hourglass_model
from hourglass.data import hourglass_dataset
from hourglass.callbacks import EvalCallBack
from common.utils import get_classes, get_matchpoints, get_model_type, optimize_tf_gpu
from common.model_utils import get_optimizer

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)


def main(args):
    log_dir = 'logs/'
    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)
    if args.matchpoint_path:
        matchpoints = get_matchpoints(args.matchpoint_path)
    else:
        matchpoints = None

    # choose model type
    if args.tiny:
        num_channels = 128
        #input_size = (192, 192)
    else:
        num_channels = 256
        #input_size = (256, 256)

    input_size = args.model_image_size

    # get train/val dataset
    train_dataset = hourglass_dataset(args.dataset_path, class_names,
                                input_size=input_size, is_train=True, matchpoints=matchpoints)

    train_gen = train_dataset.generator(args.batch_size, args.num_stacks, sigma=1, is_shuffle=True,
                                        rot_flag=True, scale_flag=True, h_flip_flag=True, v_flip_flag=True)

    model_type = get_model_type(args.num_stacks, args.mobile, args.tiny, input_size)

    # callbacks for training process
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    eval_callback = EvalCallBack(log_dir, args.dataset_path, class_names, input_size, model_type)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [tensorboard, eval_callback, terminate_on_nan]

    # prepare optimizer
    #optimizer = RMSprop(lr=5e-4)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=None)

    # get train model, doesn't specify input size
    model = get_hourglass_model(num_classes, args.num_stacks, num_channels, mobile=args.mobile)
    print('Create {} Stacked Hourglass model with stack number {}, channel number {}. train input size {}'.format('Mobile' if args.mobile else '', args.num_stacks, num_channels, input_size))
    model.summary()

    if args.weights_path:
        model.load_weights(args.weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(args.weights_path))

    # support multi-gpu training
    template_model = None
    if args.gpu_num >= 2:
        # keep the template model for saving result
        template_model = model
        model = multi_gpu_model(model, gpus=args.gpu_num)

    model.compile(optimizer=optimizer, loss=mean_squared_error)

    # start training
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_dataset.get_dataset_size() // args.batch_size,
                        epochs=args.total_epoch,
                        initial_epoch=args.init_epoch,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks=callbacks)

    if template_model is not None:
        template_model.save(os.path.join(log_dir, 'trained_final.h5'))
    else:
        model.save(os.path.join(log_dir, 'trained_final.h5'))

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument("--num_stacks", type=int, required=False, default=2,
        help='number of hourglass stacks, default=2')
    parser.add_argument("--mobile", default=False, action="store_true",
        help="use depthwise conv in hourglass'")
    parser.add_argument("--tiny", default=False, action="store_true",
        help="tiny network for speed, feature channel=128")
    parser.add_argument('--model_image_size', type=str, required=False, default='256x256',
        help = "model image input size as <num>x<num>, default 256x256")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--dataset_path', type=str, required=False, default='data/mpii',
        help='dataset path containing images and annotation file, default=data/mpii')
    parser.add_argument('--classes_path', type=str, required=False, default='configs/mpii_classes.txt',
        help='path to keypoint class definitions, default=configs/mpii_classes.txt')
    parser.add_argument('--matchpoint_path', type=str, required=False, default='configs/mpii_match_point.txt',
        help='path to matching keypoint definitions for horizontal/vertical flipping image, default=configs/mpii_match_point.txt')

    # Training options
    parser.add_argument("--batch_size", type=int, required=False, default=16,
        help='batch size for training, default=16')
    parser.add_argument('--optimizer', type=str,required=False, default='rmsprop',
        help = "optimizer for training (adam/rmsprop/sgd), default=rmsprop")
    parser.add_argument('--learning_rate', type=float,required=False, default=5e-4,
        help = "Initial learning rate, default=0.0005")
    parser.add_argument("--init_epoch", type=int, required=False, default=0,
        help="initial training epochs for fine tune training, default=0")
    parser.add_argument("--total_epoch", type=int, required=False, default=100,
        help="total training epochs, default=100")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=1')

    args = parser.parse_args()
    height, width = args.model_image_size.split('x')
    args.model_image_size = (int(height), int(width))

    main(args)
