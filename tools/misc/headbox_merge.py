#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
import scipy.io as sio


def main(args):
    anno_file = open(args.json_file)
    annotations = json.load(anno_file)

    detection_mat = sio.loadmat(args.mat_file)
    headboxes_src = detection_mat['headboxes_src']

    val_idx = 0
    for idx, item in enumerate(annotations):
        if item['isValidation'] == True:
            annotations[idx]['headboxes'] = list([list(headboxes_src[:,:,0][0]), list(headboxes_src[:,:,0][1])])
        else:
            annotations[idx]['headboxes'] = list([list([0., 0.]), list([0., 0.])])

    f = open(args.output_file, 'w')
    json.dump(annotations, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge headboxes info in val detection .mat annotation to origin MPII json annotation file, and create new json annotation file')

    parser.add_argument('--json_file', type=str, required=True, help='original json annotation file')
    parser.add_argument('--mat_file', type=str, required=True, help='val detection .mat annotation file')
    parser.add_argument('--output_file', type=str, required=False, help='output json annotation file include headbox info, default=%(default)s', default='./new_annotations.json')

    args = parser.parse_args()

    main(args)

