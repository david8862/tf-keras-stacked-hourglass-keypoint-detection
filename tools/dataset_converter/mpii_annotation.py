#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import numpy as np
import argparse, json
import scipy.io as sio

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



def main(args):
    # load MPII mat file
    print('start loading MPII MAT file...')
    mat_data = sio.loadmat(args.mat_file)
    print('Done')
    mat_data = mat_data['RELEASE']

    # get some global annotations
    annolist = mat_data['annolist'][0][0][0]
    img_train = mat_data['img_train'][0][0][0]
    single_person = mat_data['single_person'][0][0]

    #output json format annotation list
    output_list = []
    train_count = 0
    val_count = 0

    for i, annotation in enumerate(annolist):
        img_name = str(annotation['image']['name'][0][0][0])
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
            idx_list = [int(a-1) for a in range(len(rects))]


        for idx in idx_list:
            rect = rects[idx]
            if (rect is None) or check_empty(rect, 'annopoints') == True:
                # no valid annopoints
                continue

            # parse keypoints annotation
            points = rect['annopoints']['point'][0][0][0]
            points_rect = [[0., 0., 0.] for j in range(16)]
            for point in points:
                point_id = point['id'][0][0]
                x = point['x'][0][0]
                y = point['y'][0][0]
                is_visible = get_visible(point)
                points_rect[point_id] = list([float(x), float(y), float(is_visible)])

            # parse scale & objpos annotation
            scale = float(rect['scale'][0][0])
            objpos = list([float(rect['objpos']['x'][0][0][0][0]), float(rect['objpos']['y'][0][0][0][0])])

            # count train/val number
            if img_train[i]:
                train_count = train_count + 1
            else:
                val_count = val_count + 1

            #form up annotation dict item
            annotation_record = {}
            annotation_record['dataset'] = 'MPI'
            annotation_record['img_paths'] = img_name
            annotation_record['isValidation'] = float(1 - img_train[i])
            annotation_record['scale_provided'] = scale
            annotation_record['objpos'] = objpos
            annotation_record['joint_self'] = points_rect
            annotation_record['head_box'] = list([0., 0., 0., 0.])

            #parse head_box annotation
            if check_empty(rect, 'x1') == False:
                x1 = rect['x1'][0][0]
                y1 = rect['y1'][0][0]
                x2 = rect['x2'][0][0]
                y2 = rect['y2'][0][0]
                annotation_record['head_box'] = list([float(x1), float(y1), float(x2), float(y2)])

            output_list.append(annotation_record)

    print('total person number', len(output_list))
    print('train person number', train_count)
    print('val person number', val_count)
    f = open(args.output_file, 'w')
    json.dump(output_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse MPII dataset .mat annotation to our json annotation file')

    parser.add_argument('--mat_file', type=str, required=True, help='MPII mat file')
    parser.add_argument('--output_file', type=str, required=False, help='output json annotation file, default=%(default)s', default='./train_annotations.json')
    parser.add_argument('--single_only', action="store_true", help='only include single person sample', default=False)

    args = parser.parse_args()

    main(args)

