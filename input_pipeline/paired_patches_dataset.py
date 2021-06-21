# COPYRIGHT 2021. Fred Fung. Boston University.
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from input_pipeline.augmentation import Augmentation
from input_pipeline.datasets import GOTDataset
from input_pipeline.datasets import OTBDataset, LaSOTDataset, YouTubeBoundingBoxDataset
from input_pipeline.datasets import VisualGenomeDataset, MSCOCODataset
from utils.anchor import AnchorTarget
from utils.bbox import center2corner, Center, Corner
from utils.logging import get_logger

DATA_SETS = {
    "vg": VisualGenomeDataset,
    "mscoco": MSCOCODataset,
    "lasot": LaSOTDataset,
    "youtube": YouTubeBoundingBoxDataset,
    "otb": OTBDataset,
    "got": GOTDataset,
}


class PairedPatchesDataset(Dataset):
    def __init__(self, data_cfg, anchor_cfg, search_size, exemplar_size, output_size):
        self.data_cfg = data_cfg
        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.output_size = output_size
        self._logger = get_logger(__name__)
        self._logger.info("BUILDING PAIRED PATCHES DATASET...")
        templates = []
        searches = []
        xyxy_bboxes = []
        phrases = []
        for dataset in self.data_cfg.DATASETS:
            data_loader = DATA_SETS.get(dataset)(self.data_cfg)
            data_loader.prepare_dataset_for_siamese_net()
            templates += data_loader.template_files
            searches += data_loader.search_files
            xyxy_bboxes += data_loader.xyxy_boxes
            phrases += data_loader.phrases
        self._logger.info("SIZE OF THE DATASET IS: %d" % len(templates))
        self._logger.info("SHUFFLING THE ENTIRE TRAINING SET. MAY TAKE A WHILE...")
        self.list_of_datapoints = list(zip(templates, searches, xyxy_bboxes, phrases))
        self.resample(self.data_cfg.VIDEOS_PER_EPOCH)
        self._logger.info("SHUFFLING COMPLETED.")
        self._build_anchors_and_augmentations(search_size, output_size, anchor_cfg)

    def _build_anchors_and_augmentations(self, search_size, output_size, anchor_cfg):
        self.anchor_target = AnchorTarget(search_size, output_size, anchor_cfg=anchor_cfg)
        self.template_aug = Augmentation(
            self.data_cfg.TEMPLATE.SHIFT,
            self.data_cfg.TEMPLATE.SCALE,
            self.data_cfg.TEMPLATE.BLUR,
            self.data_cfg.TEMPLATE.FLIP,
            self.data_cfg.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            self.data_cfg.SEARCH.SHIFT,
            self.data_cfg.SEARCH.SCALE,
            self.data_cfg.SEARCH.BLUR,
            self.data_cfg.SEARCH.FLIP,
            self.data_cfg.SEARCH.COLOR
        )

    def resample(self, num_of_pairs):
        if len(self.list_of_datapoints) == 0:
            return
        if len(self.list_of_datapoints) < num_of_pairs:
            self._logger.warning("Sampled number of pairs (%d) is greater than the dataset size (%d)." % (
                num_of_pairs, len(self.list_of_datapoints)))
            num_of_pairs = len(self.list_of_datapoints)
        sampled_dp = random.sample(self.list_of_datapoints, num_of_pairs)
        self.template, self.search, self.xyxy_bboxes, self.phrases = zip(*sampled_dp)

    def _get_bbox(self, image, shape, exemplar_size):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return len(self.template)

    def __getitem__(self, index):
        gray = self.data_cfg.GRAY and self.data_cfg.GRAY > np.random.random()
        neg = self.data_cfg.NEG and self.data_cfg.NEG > np.random.random()
        template_image = cv2.imread(self.template[index])
        search_image = cv2.imread(self.search[index])
        template_box = Corner(0., 0., 0., 0.)
        xyxybox = Corner(self.xyxy_bboxes[index][0], self.xyxy_bboxes[index][1], self.xyxy_bboxes[index][2],
                         self.xyxy_bboxes[index][3])
        search_box = self._get_bbox(search_image, xyxybox, self.exemplar_size)

        template, _ = self.template_aug(template_image,
                                        template_box,
                                        self.exemplar_size,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       self.search_size,
                                       gray=gray)

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(bbox, self.output_size, neg)
        cls = torch.from_numpy(cls)
        delta = torch.from_numpy(delta)
        delta_weight = torch.from_numpy(delta_weight)
        bbox = torch.from_numpy(np.array(bbox))
        template = torch.from_numpy(template.transpose((2, 0, 1)).astype(np.float32))
        search = torch.from_numpy(search.transpose((2, 0, 1)).astype(np.float32))
        return {"template": template, "search": search, "label_cls": cls, "label_loc": delta,
                "label_loc_weight": delta_weight, "bbox": bbox, "phrase": self.phrases[index]}


class TestDataset(PairedPatchesDataset):
    def __init__(self, data_cfg, anchor_cfg, search_size, exemplar_size, output_size):
        self.data_cfg = data_cfg
        self.search_size = search_size
        self.exemplar_size = exemplar_size
        self.output_size = output_size
        self._logger = get_logger(__name__)
        data_cfg.DATASETS = []
        super(Dataset).__init__()
        data_loader = LaSOTDataset(data_cfg, training=False)
        data_loader.prepare_dataset_for_siamese_net()
        templates = data_loader.template_files
        searches = data_loader.search_files
        xyxy_bboxes = data_loader.xyxy_boxes
        phrases = data_loader.phrases
        self._logger.info("SIZE OF THE TEST DATASET IS: %d" % len(templates))
        self._logger.info("SHUFFLING THE ENTIRE TEST SET. MAY TAKE A WHILE...")
        self.list_of_datapoints = list(zip(templates, searches, xyxy_bboxes, phrases))
        self.resample(self.data_cfg.VIDEOS_FOR_TESTING)
        self._build_anchors_and_augmentations(search_size, output_size, anchor_cfg)
