# COPYRIGHT 2021. Fred Fung. Boston University.
r"""Utilities for inference on tracking."""
import sys
from enum import Enum

import cv2
import numpy as np
import torch
from absl import logging


class TrackingState(Enum):
    STABLE = 'STABLE'
    LOST = 'LOST'
    CONTINUED_LOST = 'CONTINUED_LOST'
    RESTORE = 'RESTORE'


def compute_difference_between(templates, new_template, margin):
    """

    :param templates: history of template patches (convolution feat map).
    :param new_template: high confidence new detection patch ready to add into the history (convolution feat map).
    :param margin: NL initialized margin
    :return:
    """
    min_distance = sys.float_info.max
    for existing_template in templates:
        distance = _compute_distance(existing_template, new_template)
        if distance < min_distance:
            min_distance = distance
    logging.info(min_distance)
    return min_distance > margin


def _compute_distance(template1, template2):
    """
    Distance between two list of convolution feature maps.
    :param template1: [(1,16,16,256),(1,16,16,256),(1,16,16,256)] Feature Map.
    :param template2: [(1,16,16,256),(1,16,16,256),(1,16,16,256)] Feature Map.
    :return:
    """
    distance = []
    for i in range(len(template1)):
        distance.append(torch.norm(template1[i] - template2[i]).cpu().numpy())
    return np.mean(distance)


def change(r):
    return np.maximum(r, 1. / r)


def compute_size(w, h):
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))


def get_subwindow_pyramid(im, pos, model_sz, size_x_scales, avg_chans):
    pyramid = [get_subwindow(im, pos, model_sz, size_x_scale, avg_chans)
               for size_x_scale in size_x_scales]
    return pyramid


def get_subwindow(im, pos, crop_size, scale_z, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    h, w, k = im.shape
    original_size = crop_size / scale_z
    c = int((original_size + 1) / 2.)
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + original_size - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + original_size - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - w + 1))
    bottom_pad = int(max(0., context_ymax - h + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (h + top_pad + bottom_pad, w + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + h, left_pad:left_pad + w, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + w, :] = avg_chans
        if bottom_pad:
            te_im[h + top_pad:, left_pad:left_pad + w, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, w + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]

    im_patch = cv2.resize(im_patch, (crop_size, crop_size))
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    im_patch = im_patch.cuda()
    return im_patch


def convert_score(score):
    score = torch.nn.functional.softmax(score[0], dim=-1)[:, :, :, 1]
    return score


def convert_bbox(delta, anchor):
    b, _, sh, sw = delta.shape
    delta = delta[0].view(4, -1, sh, sw).cpu().numpy()
    delta[0, :] = delta[0, :] * anchor[2, :] + anchor[0, :]
    delta[1, :] = delta[1, :] * anchor[3, :] + anchor[1, :]
    delta[2, :] = np.exp(delta[2, :]) * anchor[2, :]
    delta[3, :] = np.exp(delta[3, :]) * anchor[3, :]
    return delta


def bbox_clip(cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height


def log_softmax(cls):
    b, a2, h, w = cls.size()
    cls = cls.view(b, 2, a2 // 2, h, w)
    cls = cls.permute(0, 2, 3, 4, 1).contiguous()
    cls = torch.nn.functional.log_softmax(cls, dim=4)
    return cls
