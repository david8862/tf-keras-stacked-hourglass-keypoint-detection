#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data process utility functions."""
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import math

# MPII "scale" parameter use 200 pixel as person height reference
#
# http://human-pose.mpi-inf.mpg.de/#download:
#
# .scale - person scale w.r.t. 200 px height
#
MPII_SCALE_REFERENCE = 200.0


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def random_horizontal_flip(image, keypoints, center, matchpoints=None, prob=.5):
    """
    Random horizontal flip for image and keypoints

    # Arguments
        image: origin image for horizontal flip
            numpy array containing image data
        keypoints: keypoints numpy array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        center: center points array with format (x, y)
        matchpoints: list of tuple for keypoint pair index,
            which need to swap in horizontal flip
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        flip_image: fliped numpy array image.
        keypoints: fliped keypoints numpy array
        flip_center: fliped center points numpy array
    """
    flip = rand() < prob
    if not flip:
        return image, keypoints, center

    keypoints = np.copy(keypoints)

    # some keypoint pairs also need to be fliped
    # on new image
    #matchpoints = (
        #[0, 5],  # ankle
        #[1, 4],  # knee
        #[2, 3],  # hip
        #[10, 15],  # wrist
        #[11, 14],  # elbow
        #[12, 13]  # shoulder
    #)

    org_height, org_width, channels = image.shape

    # horizontal flip image: flipCode=1
    flip_image = cv2.flip(image, flipCode=1)

    # horizontal flip each keypoints
    keypoints[:, 0] = org_width - keypoints[:, 0]

    # horizontal swap matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(keypoints[i, :])
            keypoints[i, :] = keypoints[j, :]
            keypoints[j, :] = temp

    # horizontal flip center
    flip_center = center
    flip_center[0] = org_width - center[0]

    return flip_image, keypoints, flip_center


def random_vertical_flip(image, keypoints, center, matchpoints=None, prob=.5):
    """
    Random vertical flip for image and keypoints

    # Arguments
        image: origin image for vertical flip
            numpy array containing image data
        keypoints: keypoints numpy array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        center: center points array with format (x, y)
        matchpoints: list of tuple for keypoint pair index,
            which need to swap in vertical flip
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        flip_image: fliped numpy array image.
        keypoints: fliped keypoints numpy array
        flip_center: fliped center points numpy array
    """
    flip = rand() < prob
    if not flip:
        return image, keypoints, center

    keypoints = np.copy(keypoints)

    # some keypoint pairs also need to be fliped
    # on new image

    org_height, org_width, channels = image.shape

    # vertical flip image: flipCode=0
    flip_image = cv2.flip(image, flipCode=0)

    # vertical flip each keypoints
    keypoints[:, 1] = org_height - keypoints[:, 1]

    # vertical flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(keypoints[i, :])
            keypoints[i, :] = keypoints[j, :]
            keypoints[j, :] = temp

    # vertical flip center
    flip_center = center
    flip_center[1] = org_height - center[1]

    return flip_image, keypoints, flip_center


def random_brightness(image, jitter=.5):
    """
    Random adjust brightness for image

    # Arguments
        image: origin image for brightness change
            numpy array containing image data
        jitter: jitter range for random brightness,
            scalar to control the random brightness level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_bri = ImageEnhance.Brightness(img)
    brightness = rand(jitter, 1/jitter)
    new_img = enh_bri.enhance(brightness)
    image = np.asarray(new_img)

    return image


def random_chroma(image, jitter=.5):
    """
    Random adjust chroma (color level) for image

    # Arguments
        image: origin image for chroma change
            numpy array containing image data
        jitter: jitter range for random chroma,
            scalar to control the random color level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_col = ImageEnhance.Color(img)
    color = rand(jitter, 1/jitter)
    new_img = enh_col.enhance(color)
    image = np.asarray(new_img)

    return image


def random_contrast(image, jitter=.5):
    """
    Random adjust contrast for image

    # Arguments
        image: origin image for contrast change
            numpy array containing image data
        jitter: jitter range for random contrast,
            scalar to control the random contrast level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_con = ImageEnhance.Contrast(img)
    contrast = rand(jitter, 1/jitter)
    new_img = enh_con.enhance(contrast)
    image = np.asarray(new_img)

    return image


def random_sharpness(image, jitter=.5):
    """
    Random adjust sharpness for image

    # Arguments
        image: origin image for sharpness change
            numpy array containing image data
        jitter: jitter range for random sharpness,
            scalar to control the random sharpness level.

    # Returns
        image: adjusted numpy array image.
    """
    img = Image.fromarray(image)
    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = rand(jitter, 1/jitter)
    new_img = enh_sha.enhance(sharpness)
    image = np.asarray(new_img)

    return image


def random_blur(image, prob=.2, size=5):
    """
    Random add gaussian blur to image

    # Arguments
        image: origin image for blur
            numpy array containing image data
        prob: probability for blur,
            scalar to control the blur probability.
        size: kernel size for gaussian blur,
            scalar to control the filter size.

    # Returns
        image: adjusted numpy array image.
    """
    blur = rand() < prob
    if blur:
        image = cv2.GaussianBlur(image, (size, size), 0)

    return image


def random_histeq(image, size=8, prob=.2):
    """
    Random apply "Contrast Limited Adaptive Histogram Equalization"
    to image

    # Arguments
        image: origin image for histeq
            numpy array containing image data
        size: grid size for CLAHE,
            scalar to control the grid size.
        prob: probability for histeq,
            scalar to control the histeq probability.

    # Returns
        image: adjusted numpy array image.
    """
    histeq = rand() < prob
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
    return image


def random_grayscale(image, prob=.1):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            numpy array containing image data
        prob: probability for grayscale convert,
            scalar to control the convert probability.

    # Returns
        image: adjusted numpy array image.
    """
    convert = rand() < prob
    if convert:
        #convert to grayscale first, and then
        #back to 3 channels fake BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def random_rotate_angle(rotate_range, prob=0.5):
    """
    Random rotate angle for image and keypoints transform

    # Arguments
        rotate_range: jitter range for random rotate,
            scalar to control the random rotate level.
        prob: probability for rotate transform,
            scalar to control the rotate probability.

    # Returns
        rotate_angle: a rotate angle value.
    """
    rotate = rand() < prob
    if not rotate:
        return 0

    rotate_angle = np.random.randint(-1*rotate_range, rotate_range)
    return rotate_angle


#######################################################################################################
# 2 solutions of image & keypoints transform


###############################
# Option 1 (from origin repo):
def get_transform(center, scale, shape, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = scale * MPII_SCALE_REFERENCE
    t = np.zeros((3, 3))
    t[0, 0] = float(shape[1]) / h
    t[1, 1] = float(shape[0]) / h
    t[0, 2] = shape[1] * (-float(center[0]) / h + .5)
    t[1, 2] = shape[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -shape[1] / 2
        t_mat[1, 2] = -shape[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, shape, invert=0, rot=0):
    """
    Transform pixel location to different reference
    """
    t = get_transform(center, scale, shape, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop_image(img, center, scale, shape, rotate_angle=0):
    """
    Crop out single person area from image with center point and scale factor,
    together with rotate and resize to model input size
    """
    # preprocessing for efficient cropping
    height, width = img.shape[0:2]
    scale_factor = scale * MPII_SCALE_REFERENCE / shape[0]
    if scale_factor < 2:
        scale_factor = 1
    else:
        new_size = int(np.math.floor(max(height, width) / scale_factor))
        new_height = int(np.math.floor(height / scale_factor))
        new_width = int(np.math.floor(width / scale_factor))
        img = np.array(Image.fromarray(img).resize((new_width, new_height), Image.BICUBIC))
        center = center * 1.0 / scale_factor
        scale = scale / scale_factor

    # upper left point
    upper_left = np.array(transform([0, 0], center, scale, shape, invert=1))
    # bottom right point
    bottom_right = np.array(transform(shape, center, scale, shape, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(bottom_right - upper_left) / 2 - float(bottom_right[1] - upper_left[1]) / 2)
    if not rotate_angle == 0:
        upper_left -= pad
        bottom_right += pad

    new_shape = [bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -upper_left[0]), min(bottom_right[0], len(img[0])) - upper_left[0]
    new_y = max(0, -upper_left[1]), min(bottom_right[1], len(img)) - upper_left[1]
    # Range to sample from original image
    old_x = max(0, upper_left[0]), min(len(img[0]), bottom_right[0])
    old_y = max(0, upper_left[1]), min(len(img), bottom_right[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rotate_angle == 0:
        # Remove padding
        new_img = np.array(Image.fromarray(new_img).rotate(rotate_angle))
        new_img = new_img[pad:-pad, pad:-pad]

    if 0 in new_img.shape:
        # in case we got a empty image
        return None

    new_img = np.array(Image.fromarray(new_img).resize(tuple(reversed(shape)), Image.BICUBIC))
    return new_img


def transform_keypoints(keypoints, center, scale, shape, rotate_angle):
    """
    Transform keypoints to single person image reference
    """
    new_keypoints = np.copy(keypoints)
    for i in range(keypoints.shape[0]):
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
            _x = transform(new_keypoints[i, 0:2] + 1, center=center, scale=scale, shape=shape, invert=0, rot=rotate_angle)
            new_keypoints[i, 0:2] = _x
    return new_keypoints


def invert_transform_keypoints(keypoints, center, scale, shape, rotate_angle):
    """
    Inverted transform keypoints back to origin image reference
    """
    new_keypoints = np.copy(keypoints)
    for i in range(keypoints.shape[0]):
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
            _x = transform(new_keypoints[i, 0:2] + 1, center=center, scale=scale, shape=shape, invert=1, rot=rotate_angle)
            new_keypoints[i, 0:2] = _x
    return new_keypoints

# End of Option 1
###############################


###############################
# Option 2:
def revert_keypoints(keypoints, center, scale, image_shape, input_shape, output_stride=4):
    """
    Revert transform/predict keypoints back to origin image reference,
    used for val/eval dataset, so we didn't support any augment like flip
    or rotate
    """
    height, width, channels = image_shape

    person_height = scale * MPII_SCALE_REFERENCE
    # get person width same aspect ratio as target shape
    person_width = person_height * (float(input_shape[1]) / float(input_shape[0]))

    # person bbox
    person_xmin = int(max(0, center[0] - (person_width // 2)))
    person_xmax = int(min(width, center[0] + (person_width // 2)))
    person_ymin = int(max(0, center[1] - (person_height // 2)))
    person_ymax = int(min(height, center[1] + (person_height // 2)))

    # calculate actual resize ratio on width and height
    crop_height = person_ymax - person_ymin
    crop_width = person_xmax - person_xmin
    resize_ratio_x = input_shape[1] / float(crop_width)
    resize_ratio_y = input_shape[0] / float(crop_height)

    # update keypoints to single person reference
    new_keypoints = np.zeros_like(keypoints)

    for i in range(keypoints.shape[0]):
        # only pick valid keypoint
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
            # move and resize the keypoint
            new_x = min(width, (keypoints[i, 0] * output_stride / resize_ratio_x) + person_xmin)
            new_y = min(height, (keypoints[i, 1] * output_stride / resize_ratio_y) + person_ymin)

            # only pick valid new keypoint
            if new_x < width and new_y < height:
                new_keypoints[i, 0:2] = np.asarray([new_x, new_y])
                new_keypoints[i, 2] = keypoints[i, 2]
    return new_keypoints



def crop_single_person(image, keypoints, center, scale, input_shape):
    """
    crop out single person area from origin image with center point &
    scale factor, and resize to model input size
    """
    height, width, channels = image.shape

    person_height = scale * MPII_SCALE_REFERENCE
    # get person width same aspect ratio as target shape
    person_width = person_height * (float(input_shape[1]) / float(input_shape[0]))

    # person bbox
    person_xmin = int(max(0, center[0] - (person_width // 2)))
    person_xmax = int(min(width, center[0] + (person_width // 2)))
    person_ymin = int(max(0, center[1] - (person_height // 2)))
    person_ymax = int(min(height, center[1] + (person_height // 2)))

    # crop out person image area
    person_image = np.copy(image[person_ymin:person_ymax, person_xmin:person_xmax])

    # resize person image to target shape
    person_image = cv2.resize(person_image, tuple(reversed(input_shape)), cv2.INTER_AREA)

    # calculate actual resize ratio on width and height
    crop_height = person_ymax - person_ymin
    crop_width = person_xmax - person_xmin
    resize_ratio_x = input_shape[1] / float(crop_width)
    resize_ratio_y = input_shape[0] / float(crop_height)

    # update keypoints to single person reference
    new_keypoints = np.zeros_like(keypoints)

    for i in range(keypoints.shape[0]):
        # only pick valid keypoint
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
            # move and resize the keypoint
            new_x = (keypoints[i, 0] - person_xmin) * resize_ratio_x
            new_y = (keypoints[i, 1] - person_ymin) * resize_ratio_y

            # only pick valid new keypoint
            if new_x > 0 and new_y > 0:
                new_keypoints[i, 0:2] = np.asarray([new_x, new_y])
                new_keypoints[i, 2] = 1.0

    return person_image, new_keypoints


def rotate_single_person(image, keypoints, angle):
    """
    rotate single person image and keypoints coordinates
    """
    # image center point for rotation
    center_x, center_y = image.shape[1]//2, image.shape[0]//2

    # convert angle to randian
    radian = math.radians(angle)

    # rotate image
    # getRotationMatrix2D() follow counterclockwise rotation, so here we
    # use "-1*angle" to align with following keypoint rotation
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), -1*angle, scale=1.0)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # rotate keypoints (clockwise rotation)
    new_keypoints = np.zeros_like(keypoints)
    for i in range(keypoints.shape[0]):
        # only pick valid keypoint
        if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
            new_x = (keypoints[i, 0] - center_x) * math.cos(radian) - (keypoints[i, 1] - center_y) * math.sin(radian) + center_x
            new_y = (keypoints[i, 0] - center_x) * math.sin(radian) + (keypoints[i, 1] - center_y) * math.cos(radian) + center_y

            # only pick valid new keypoint
            if (new_x > 0 and new_x < image.shape[1]) and (new_y > 0 and new_y < image.shape[0]):
                new_keypoints[i, 0:2] = np.asarray([new_x, new_y])
                new_keypoints[i, 2] = 1.0

    return image, new_keypoints

# End of Option 2
###############################


def label_heatmap(img, pt, sigma, type='Gaussian'):
    """
    Create a 2D gaussian label heatmap
    Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    Check that any part of the gaussian is in-bounds
    """
    upper_left = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    bottom_right = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (upper_left[0] < 0 or upper_left[1] < 0 or
            bottom_right[0] >= img.shape[1] or bottom_right[1] >= img.shape[0]):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = [max(0, -upper_left[0]), min(bottom_right[0], img.shape[1]) - upper_left[0]]
    g_y = [max(0, -upper_left[1]), min(bottom_right[1], img.shape[0]) - upper_left[1]]
    # Image range
    img_x = [max(0, upper_left[0]), min(bottom_right[0], img.shape[1])]
    img_y = [max(0, upper_left[1]), min(bottom_right[1], img.shape[0])]

    # NOTE: an ugly trick to avoid heatmap size mismatch,
    # which is usually caused by python/numpy calculate error
    if img_x[1] - img_x[0] > g.shape[0]:
        img_x[1] -= (img_x[1] - img_x[0]) - g.shape[0]

    if img_y[1] - img_y[0] > g.shape[0]:
        img_y[1] -= (img_y[1] - img_y[0]) - g.shape[0]

    # Apply heatmap to image
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def generate_gt_heatmap(keypoints, heatmap_size, sigma=1):
    """
    generate ground truth keypoints heatmap

    # Arguments
        keypoints: keypoints array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        heatmap_size: ground truth heatmap shape
            numpy array containing segment label mask
        sigma: variance of the 2D gaussian heatmap distribution

    # Returns
        gt_heatmap: ground truth keypoints heatmap,
                    shape=(heatmap_size[0], heatmap_size[1], num_keypoints)
    """
    num_keypoints = keypoints.shape[0]
    gt_heatmap = np.zeros(shape=(heatmap_size[0], heatmap_size[1], num_keypoints), dtype=float)

    for i in range(num_keypoints):
        visibility = keypoints[i, 2]
        if visibility > 0:
            # only apply heatmap when visibility = 1.0
            gt_heatmap[:, :, i] = label_heatmap(gt_heatmap[:, :, i], keypoints[i, :], sigma)

    return gt_heatmap


def normalize_image(imgdata, color_mean):
    '''
    :param imgdata: image in 0 ~ 255
    :return:  image from 0.0 to 1.0
    '''
    imgdata = imgdata / 255.0

    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    return imgdata


def denormalize_image(imgdata, color_mean):
    '''
    :param imgdata: image from 0.0 to 1.0
    :return:  image in 0 ~ 255
    '''
    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] += color_mean[i]

    imgdata = (imgdata*255.0).astype(np.uint8)

    return imgdata


def preprocess_image(image, model_input_size, mean=(0.4404, 0.4440, 0.4327)):
    """
    Prepare model input image data with
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(tuple(reversed(model_input_size)), Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')

    mean = np.array(mean, dtype=np.float)
    image_data = normalize_image(image_data, mean)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image_data
