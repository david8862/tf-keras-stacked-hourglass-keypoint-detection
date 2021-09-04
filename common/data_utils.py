#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data process utility functions."""
import numpy as np
from PIL import Image, ImageEnhance
import cv2


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def random_horizontal_flip(image, joints, center, matchpoints=None, prob=.5):
    """
    Random horizontal flip for image and keypoints

    # Arguments
        image: origin image for horizontal flip
            numpy array containing image data
        joints: keypoints numpy array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        center: center points array with format (x, y)
        matchpoints: list of tuple for keypoint pair index,
            which need to swap in horizontal flip
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        flip_image: fliped numpy array image.
        joints: fliped keypoints numpy array
        flip_center: fliped center points numpy array
    """
    flip = rand() < prob
    if not flip:
        return image, joints, center

    joints = np.copy(joints)

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

    # horizontal flip each joints
    joints[:, 0] = org_width - joints[:, 0]

    # horizontal swap matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # horizontal flip center
    flip_center = center
    flip_center[0] = org_width - center[0]

    return flip_image, joints, flip_center


def random_vertical_flip(image, joints, center, matchpoints=None, prob=.5):
    """
    Random vertical flip for image and keypoints

    # Arguments
        image: origin image for vertical flip
            numpy array containing image data
        joints: keypoints numpy array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        center: center points array with format (x, y)
        matchpoints: list of tuple for keypoint pair index,
            which need to swap in vertical flip
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        flip_image: fliped numpy array image.
        joints: fliped keypoints numpy array
        flip_center: fliped center points numpy array
    """
    flip = rand() < prob
    if not flip:
        return image, joints, center

    joints = np.copy(joints)

    # some keypoint pairs also need to be fliped
    # on new image

    org_height, org_width, channels = image.shape

    # vertical flip image: flipCode=0
    flip_image = cv2.flip(image, flipCode=0)

    # vertical flip each joints
    joints[:, 1] = org_height - joints[:, 1]

    # vertical flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # vertical flip center
    flip_center = center
    flip_center[1] = org_height - center[1]

    return flip_image, joints, flip_center


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



def get_transform(center, scale, shape, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
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
    scale_factor = scale * 200.0 / shape[0]
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


def transform_keypoints(joints, center, scale, shape, rotate_angle):
    """
    Transform keypoints to single person image reference
    """
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, shape=shape, invert=0, rot=rotate_angle)
            newjoints[i, 0:2] = _x
    return newjoints


def invert_transform_keypoints(joints, center, scale, shape, rotate_angle):
    """
    Inverted transform keypoints back to origin image reference
    """
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, shape=shape, invert=1, rot=rotate_angle)
            newjoints[i, 0:2] = _x
    return newjoints


def label_heatmap(img, pt, sigma, type='Gaussian'):
    """
    Create a 2D gaussian label heatmap
    Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    Check that any part of the gaussian is in-bounds
    """
    upper_left = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    bottom_right = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (upper_left[0] >= img.shape[1] or upper_left[1] >= img.shape[0] or
            bottom_right[0] < 0 or bottom_right[1] < 0):
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
    g_x = max(0, -upper_left[0]), min(bottom_right[0], img.shape[1]) - upper_left[0]
    g_y = max(0, -upper_left[1]), min(bottom_right[1], img.shape[0]) - upper_left[1]
    # Image range
    img_x = max(0, upper_left[0]), min(bottom_right[0], img.shape[1])
    img_y = max(0, upper_left[1]), min(bottom_right[1], img.shape[0])

    # Apply heatmap to image
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def generate_gt_heatmap(joints, heatmap_size, sigma=1):
    """
    generate ground truth keypoints heatmap

    # Arguments
        joints: keypoints array, shape=(num_keypoints, 3)
            each keypoints with format (x, y, visibility)
        heatmap_size: ground truth heatmap shape
            numpy array containing segment label mask
        sigma: variance of the 2D gaussian heatmap distribution

    # Returns
        gt_heatmap: ground truth keypoints heatmap,
                    shape=(heatmap_size[0], heatmap_size[1], num_keypoints)
    """
    num_keypoints = joints.shape[0]
    gt_heatmap = np.zeros(shape=(heatmap_size[0], heatmap_size[1], num_keypoints), dtype=float)

    for i in range(num_keypoints):
        visibility = joints[i, 2]
        if visibility > 0:
            # only apply heatmap when visibility = 1.0
            gt_heatmap[:, :, i] = label_heatmap(gt_heatmap[:, :, i], joints[i, :], sigma)

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
