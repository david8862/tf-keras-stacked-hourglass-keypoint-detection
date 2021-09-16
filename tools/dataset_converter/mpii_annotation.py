#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse, json
import scipy.io
from tqdm import tqdm

#
# MPII human pose dataset have 16 skeleton keypoints
#
MPII_KEYPOINT_NUM = 16


def check_empty(item_list, name):
    try:
        item_list[name]
    except ValueError:
        return True

    if len(item_list[name]) > 0:
        return False
    else:
        return True


def get_visible(point):
    if check_empty(point, 'is_visible') == True:
        return 0

    visible_array = point['is_visible']

    if visible_array.shape == (1,1):
        is_visible = visible_array[0][0]
    elif visible_array.shape == (1,):
        is_visible = visible_array[0]
    else:
        is_visible = 0

    return is_visible



def mpii_annotation(args):
    # load MPII mat file
    print('start loading MPII MAT file...')
    mat_data = scipy.io.loadmat(args.mat_file)
    print('Done')
    mat_data = mat_data['RELEASE']

    # get some global annotations
    annolist = mat_data['annolist'][0][0][0]
    img_train = mat_data['img_train'][0][0][0]
    single_person = mat_data['single_person'][0][0]

    #output json format annotation list
    output_list = []
    train_val_image_count = 0
    test_image_count = 0
    train_person_count = 0
    val_person_count = 0

    pbar = tqdm(total=len(annolist), desc='MPII annotation')
    for i, annotation in enumerate(annolist):
        pbar.update(1)
        img_name = str(annotation['image']['name'][0][0][0])

        # count train/test image number
        # NOTE: there's no keypoints info for test images in MPII MAT annotation
        if img_train[i] == 1:
            train_val_image_count = train_val_image_count + 1
        else:
            test_image_count = test_image_count + 1
            continue

        if check_empty(annotation, 'annorect') == True:
            # no valid annorect
            continue

        rects = annotation['annorect'][0]

        if args.single_only:
            # parse single person id in image
            single_array = single_person[i][0]
            if 0 in single_array.shape:
                continue
            idx_list = [int(a-1) for a in single_array]
        else:
            idx_list = [int(a) for a in range(len(rects))]


        for idx in idx_list:
            rect = rects[idx]
            if (rect is None) or check_empty(rect, 'annopoints') == True:
                # no valid annopoints
                continue

            # parse keypoints annotation
            points = rect['annopoints']['point'][0][0][0]
            points_rect = [[0., 0., 0.] for j in range(MPII_KEYPOINT_NUM)]
            for point in points:
                point_id = point['id'][0][0]
                x = point['x'][0][0]
                y = point['y'][0][0]
                is_visible = get_visible(point)
                points_rect[point_id] = list([float(x), float(y), float(is_visible)])

            # parse scale & objpos annotation
            scale = float(rect['scale'][0][0])
            objpos = list([float(rect['objpos']['x'][0][0][0][0]), float(rect['objpos']['y'][0][0][0][0])])

            # assign train/val according to val_split
            is_validation = float(np.random.rand() < args.val_split)

            if is_validation == 1.0:
                val_person_count = val_person_count + 1
            else:
                train_person_count = train_person_count + 1

            #form up annotation dict item
            annotation_record = {}
            annotation_record['dataset'] = 'MPI'
            annotation_record['img_paths'] = img_name
            annotation_record['isValidation'] = is_validation
            annotation_record['scale_provided'] = scale
            annotation_record['objpos'] = objpos
            annotation_record['joint_self'] = points_rect
            annotation_record['headboxes'] = list([[0.0, 0.0], [0.0, 0.0]])

            #parse headboxes annotation
            if check_empty(rect, 'x1') == False:
                x1 = rect['x1'][0][0]
                y1 = rect['y1'][0][0]
                x2 = rect['x2'][0][0]
                y2 = rect['y2'][0][0]
                annotation_record['headboxes'] = list([[float(x1), float(y1)], [float(x2), float(y2)]])

            output_list.append(annotation_record)

    pbar.close()
    print('train & val image number', train_val_image_count)
    print('test image number', test_image_count)
    print('train person number', train_person_count)
    print('val person number', val_person_count)
    print('total person number', len(output_list))
    f = open(args.output_file, 'w')
    json.dump(output_list, f)


def main():
    parser = argparse.ArgumentParser(description='Parse MPII dataset .mat annotation to our json annotation file')

    parser.add_argument('--mat_file', type=str, required=True, help='MPII mat file')
    parser.add_argument('--output_file', type=str, required=False, help='output json annotation file, default=%(default)s', default='./train_annotations.json')
    parser.add_argument('--val_split', type=float, required=False, help='validation data persentage in dataset, default=%(default)s', default=0.1)
    parser.add_argument('--single_only', action="store_true", help='only include single person sample', default=False)

    args = parser.parse_args()

    mpii_annotation(args)


if __name__ == '__main__':
    main()
