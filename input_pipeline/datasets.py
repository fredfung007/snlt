# COPYRIGHT 2021. Fred Fung. Boston University.
r"""A generic data loader for images and videos with language annotation."""
import abc
import json
import os
import random

import numpy as np
from tqdm import tqdm


class BaseDataset(abc.ABC):

    def __init__(self, dataset_cfg):
        self.template_files = []
        self.search_files = []
        self.xyxy_boxes = []
        self.phrases = []
        self.dataset_cfg = dataset_cfg

    @abc.abstractmethod
    def prepare_dataset_for_siamese_net(self):
        pass


class VisualGenomeDataset(BaseDataset):
    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.VISUAL_GENOME_JSON_FILE
        self.vg_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.VISUAL_GENOME_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as vg_json_file:
            vg_crops = json.load(vg_json_file)
        for image in tqdm(vg_crops, desc="Reading Visual Genome Images"):
            for region in vg_crops[image]:
                x_file = os.path.join(self.vg_home, "crop", image.split("/")[-1],
                                      "000000.%s.x.jpg" % region)
                z_file = os.path.join(self.vg_home, "crop", image.split("/")[-1],
                                      "000000.%s.z.jpg" % region)
                phrase = vg_crops[image][region]["phrase"].replace(".", "")
                box = np.array(vg_crops[image][region]["000000"])
                self.template_files.append(z_file)
                self.search_files.append(x_file)
                self.phrases.append(phrase)
                self.xyxy_boxes.append(box)


class MSCOCODataset(BaseDataset):
    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.MSCOCO_JSON_FILE
        self.mscoco_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.MSCOCO_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as mscoco_json_file:
            coco_crops = json.load(mscoco_json_file)
        for image in tqdm(coco_crops, desc="Reading MSCOCO Images"):
            for region in coco_crops[image]:
                x_file = os.path.join(self.mscoco_home, "crop511", image, "000000.%s.x.jpg" % region)
                z_file = os.path.join(self.mscoco_home, "crop511", image, "000000.%s.z.jpg" % region)
                box = np.array(coco_crops[image][region]["000000"])
                self.template_files.append(z_file)
                self.search_files.append(x_file)
                self.xyxy_boxes.append(box)
                self.phrases.append("")


class LaSOTDataset(BaseDataset):
    def __init__(self, dataset_cfg, training=True):
        super().__init__(dataset_cfg)
        if training:
            self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.LASOT_JSON_FILE
        else:
            self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.LASOT_TEST_JSON_FILE
        self.lasot_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.LASOT_CROP_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as lasot_json_file:
            lasot_crops = json.load(lasot_json_file)

        for video in tqdm(lasot_crops, desc="Reading LaSOT Images"):
            video_name = video.split("/")[-2] + "/" + video.split("/")[-1]
            frames = lasot_crops[video]["00"]
            for _ in range(self.dataset_cfg.LASOT_NUM_PER_VIDEO):
                frame1 = random.sample(list(frames)[:-1], 1)[0]
                frame2 = "%06d" % max(int((int(frame1) - random.randint(0, 50)) / 10) * 10, 0)
                if frame2 not in lasot_crops[video]["00"]:
                    frame2 = frame1
                if self.dataset_cfg.ALWAYS_USE_FIRST_FRAME_Z:
                    frame2 = "000000"
                x_file = os.path.join(self.lasot_home, video_name, "%s.00.x.jpg" % frame1)
                z_file = os.path.join(self.lasot_home, video_name, "%s.00.z.jpg" % frame2)
                box = np.array(lasot_crops[video]["00"][frame1])
                self.template_files.append(z_file)
                self.search_files.append(x_file)
                self.xyxy_boxes.append(box)
                phrase = lasot_crops[video]["00"]["phrase"].replace(".", "")
                self.phrases.append(phrase)


class OTBDataset(BaseDataset):
    """Data loader for OTB dataset. For inference only. No training code provided."""

    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.OTB_JSON_FILE
        self.otb_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.OTB_CROP_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as otb_json_file:
            otb_crops = json.load(otb_json_file)

        for video in tqdm(otb_crops, desc="Reading OTB Images"):
            video_name = video.split("/")[-1]
            frames = otb_crops[video]["00"]
            for _ in range(self.dataset_cfg.OTB_NUM_PER_VIDEO):
                frame1 = random.sample(list(frames)[:-1], 1)[0]
                frame2 = "%06d" % max(int(frame1) - random.randint(0, 50), 0)
                if self.dataset_cfg.ALWAYS_USE_FIRST_FRAME_Z:
                    frame2 = "000000"
                x_file = os.path.join(self.otb_home, video_name, "%s.00.x.jpg" % frame1)
                z_file = os.path.join(self.otb_home, video_name, "%s.00.z.jpg" % frame2)
                box = np.array(otb_crops[video]["00"][frame1])
                self.template_files.append(z_file)
                self.search_files.append(x_file)
                self.xyxy_boxes.append(box)
                phrase = otb_crops[video]["00"]["phrase"]
                self.phrases.append(phrase)


class YouTubeBoundingBoxDataset(BaseDataset):
    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.YOUTUBE_JSON_FILE
        self.youtube_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.YOUTUBE_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as youtube_json_file:
            youtube_crops = json.load(youtube_json_file)

        for video in tqdm(youtube_crops, desc="Reading YouTube Images"):
            if "00" not in youtube_crops[video]:
                continue
            frames = youtube_crops[video]["00"]
            if len(frames) < 2:
                continue
            count = 0
            while count < self.dataset_cfg.YOUTUBE_NUM_PER_VIDEO:
                frames = random.sample(list(frames), 2)
                frame1 = frames[0]
                frame2 = frames[1]
                x_file = os.path.join(self.youtube_home, video, "%s.00.x.jpg" % frame1)
                z_file = os.path.join(self.youtube_home, video, "%s.00.z.jpg" % frame2)
                box = np.array(youtube_crops[video]["00"][frame1])
                if os.path.isfile(x_file) and os.path.isfile(z_file) and box[2] > box[0] and box[3] > box[1]:
                    self.template_files.append(z_file)
                    self.search_files.append(x_file)
                    self.xyxy_boxes.append(box)
                    self.phrases.append("")
                count += 1


class GOTDataset(BaseDataset):
    def __init__(self, dataset_cfg):
        super().__init__(dataset_cfg)
        self.json_file = self.dataset_cfg.DATA_HOME + self.dataset_cfg.GOT_JSON_FILE
        self.got_home = self.dataset_cfg.DATA_HOME + self.dataset_cfg.GOT_HOME

    def prepare_dataset_for_siamese_net(self):
        with open(self.json_file) as got_json_file:
            got_crops = json.load(got_json_file)

        for video in tqdm(got_crops, desc="Reading GOT-10K Images"):
            if "00" not in got_crops[video]:
                continue
            frames = got_crops[video]["00"]
            if len(frames) < 2:
                continue
            count = 0
            while count < self.got_home.GOT_NUM_PER_VIDEO:
                frames = random.sample(list(frames), 2)
                frame1 = frames[0]
                frame2 = frames[1]
                x_file = os.path.join(self.got_home, video, "%s.00.x.jpg" % frame1)
                z_file = os.path.join(self.got_home, video, "%s.00.z.jpg" % frame2)
                box = np.array(got_crops[video]["00"][frame1])
                if os.path.isfile(x_file) and os.path.isfile(z_file) and box[2] > box[0] and box[3] > box[1]:
                    self.template_files.append(z_file)
                    self.search_files.append(x_file)
                    self.xyxy_boxes.append(box)
                    self.phrases.append("")
                count += 1
