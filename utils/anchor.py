# COPYRIGHT 2021. Fred Fung. Boston University.

# ADAPTED FROM Copyright (c) SenseTime. All Rights Reserved.


import math

import numpy as np

from configs import get_default_anchor_configs
from utils.bbox import IoU, corner2center, center2corner
from utils.logging import get_logger

_logger = get_logger(__name__)


class Anchors:
    """
    This class generate anchors.
    """

    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True


class AnchorTarget:
    def __init__(self, search_size, output_size, anchor_cfg=None):
        if anchor_cfg is None:
            anchor_cfg = get_default_anchor_configs()
        self.anchors = Anchors(anchor_cfg.STRIDE, anchor_cfg.RATIOS, anchor_cfg.SCALES)
        self.anchors.generate_all_anchors(im_c=search_size // 2, size=output_size)
        self.anchor_cfg = anchor_cfg
        self.search_size = search_size
        self.output_size = output_size

    def __call__(self, target, size, neg=False, is_fc=False):
        tcx, tcy, tw, th = corner2center(target)
        anchor_num = len(self.anchor_cfg.RATIOS) * len(self.anchor_cfg.SCALES)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - self.search_size // 2) / self.anchor_cfg.STRIDE + 0.5))
            cy += int(np.ceil((tcy - self.search_size // 2) / self.anchor_cfg.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.anchor_cfg.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        with np.errstate(invalid='raise'):
            try:
                delta[2] = np.log(tw / w)
                delta[3] = np.log(th / h)
            except Exception:
                _logger.warning('Invalid value encountered in log. [ANCHOR TARGET ENCODING]')
                delta[2] = np.zeros_like(w)
                delta[3] = np.zeros_like(h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.anchor_cfg.THR_HIGH)
        neg = np.where(overlap < self.anchor_cfg.THR_LOW)

        pos, pos_num = select(pos, self.anchor_cfg.POS_NUM)
        neg, neg_num = select(neg, self.anchor_cfg.TOTAL_NUM - self.anchor_cfg.POS_NUM)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap
