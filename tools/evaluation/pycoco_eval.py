#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt to evaluate COCO AP with pycocotools
'''
import os, argparse, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def pycoco_eval(annotation_file, result_file):
    cocoGt=COCO(annotation_file)
    cocoDt=cocoGt.loadRes(result_file)
    imgIds=sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    parser = argparse.ArgumentParser(description='evaluate COCO AP with pycocotools')
    parser.add_argument('--coco_result_json', required=True, type=str, help='coco json result file')
    parser.add_argument('--coco_annotation_json', required=True, type=str, help='coco json annotation file')
    args = parser.parse_args()

    pycoco_eval(args.coco_annotation_json, args.coco_result_json)


if __name__ == "__main__":
    main()
