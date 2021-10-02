#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test data argument process
"""
import os, sys, argparse
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from hourglass.data import hourglass_dataset, HG_OUTPUT_STRIDE
from common.data_utils import denormalize_image
from common.utils import get_classes, get_skeleton, render_skeleton


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Test tool for data augment process')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing images and annotation file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to keypoint class definition file')
    parser.add_argument('--skeleton_path', type=str, required=False, help='path to keypoint skeleton definitions, default None', default=None)

    parser.add_argument('--output_path', type=str, required=False,  help='output path for augmented images, default=%(default)s', default='./test')
    parser.add_argument('--batch_size', type=int, required=False, help = "batch size for test data, default=%(default)s", default=16)
    parser.add_argument('--model_input_shape', type=str, required=False, help='model image input shape as <height>x<width>, default=%(default)s', default='256x256')

    args = parser.parse_args()

    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None
    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    os.makedirs(args.output_path, exist_ok=True)

    # prepare train dataset (having augmented data process)
    data_generator = hourglass_dataset(args.dataset_path, batch_size=1, class_names=class_names, input_shape=model_input_shape, num_hgstack=1, is_train=True, with_meta=True)

    pbar = tqdm(total=args.batch_size, desc='Generate augment image')
    for i, (image_data, gt_heatmap, metainfo) in enumerate(data_generator):
        if i >= args.batch_size:
            break
        pbar.update(1)

        # get ground truth keypoints (transformed)
        metainfo = metainfo[0]
        image = image_data[0]
        gt_keypoints = metainfo['tpts']

        #un-normalize image
        image = denormalize_image(image, data_generator.get_color_mean())

        # form up gt keypoints dict
        gt_keypoints_dict = {}
        for j, keypoint in enumerate(gt_keypoints):
            gt_keypoints_dict[class_names[j]] = (keypoint[0]*HG_OUTPUT_STRIDE, keypoint[1]*HG_OUTPUT_STRIDE, 1.0)

        # render gt keypoints skeleton on image
        image_array = render_skeleton(image, gt_keypoints_dict, skeleton_lines)

        # save rendered image
        image = Image.fromarray(image_array)
        # here we handle the RGBA image
        if(len(image.split()) == 4):
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
        image.save(os.path.join(args.output_path, str(i)+".jpg"))
    pbar.close()
    print('Done. augment images have been saved in', args.output_path)


if __name__ == "__main__":
    main()

