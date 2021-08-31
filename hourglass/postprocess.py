#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def non_max_supression(plain, windowSize=3, conf_threshold=1e-6):
    # clear value less than conf_threshold
    under_th_indices = plain < conf_threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))


def post_process_heatmap(heatmap, conf_threshold=1e-6):
    """
    normal approach of keypoints heatmap post process:
      1. blur heatmap with gaussian filter
      2. do NMS with 3x3 max filter to get peak point
      3. choose max peak point as keypoint output
    """
    keypoint_list = list()
    for i in range(heatmap.shape[-1]):
        _map = heatmap[:, :, i]
        # do a heatmap blur with gaussian_filter
        _map = gaussian_filter(_map, sigma=0.5)
        # get peak point in heatmap with 3x3 max filter
        _nmsPeaks = non_max_supression(_map, windowSize=3, conf_threshold=conf_threshold)

        # choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
        # and get its coordinate & confidence
        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            keypoint_list.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            keypoint_list.append((0, 0, 0))
    return keypoint_list


def post_process_heatmap_simple(heatmap, conf_threshold=1e-6):
    """
    A simple approach of keypoints heatmap post process,
    only pick 1 max point in each heatmap as keypoint output
    """
    keypoint_list = list()
    for i in range(heatmap.shape[-1]):
        # ignore last channel, background channel
        _map = heatmap[:, :, i]
        # clear value less than conf_threshold
        under_th_indices = _map < conf_threshold
        _map[under_th_indices] = 0

        # choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
        # and get its coordinate & confidence
        y, x = np.where(_map == _map.max())
        if len(x) > 0 and len(y) > 0:
            keypoint_list.append((int(x[0]), int(y[0]), _map[y[0], x[0]]))
        else:
            keypoint_list.append((0, 0, 0))
    return keypoint_list

