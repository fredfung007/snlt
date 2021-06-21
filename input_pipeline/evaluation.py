# COPYRIGHT 2021. Fred Fung. Boston University.
import os

import numpy as np


class TNL2KDataLoader:
    def __init__(self, tnl2k_home):
        self.tnl2k_home = tnl2k_home
        self.test_video_names = []
        for video in os.listdir(self.tnl2k_home):
            if video != "NL":
                self.test_video_names.append(video)

    def get_next_batch(self, video_name):
        path_to_video = os.path.join(self.tnl2k_home, video_name)
        with open(os.path.join(path_to_video, "groundtruth.txt")) as gt_boxes_file:
            all_gt_boxes = gt_boxes_file.readlines()

        with open(os.path.join(self.tnl2k_home, "NL", "%s_language.txt" % video_name)) as gt_nlp_file:
            phrases = gt_nlp_file.readline().strip()

        image_files = os.listdir(path_to_video + "/imgs/")

        length = len(all_gt_boxes)
        images = []
        gt_boxes = []
        for frame_num in range(length):
            gt_box = all_gt_boxes[frame_num]
            gt_box = np.array([int(index) for index in gt_box.split(",")], dtype=np.float32)
            images.append(path_to_video + "/imgs/" + image_files[frame_num])
            gt_boxes.append(gt_box)
        return video_name, phrases, images, np.stack(gt_boxes, axis=0)


class LaSOTDataLoader:
    def __init__(self, lasot_test_dataset_file, lasot_home):
        with open(lasot_test_dataset_file) as test_dataset:
            self.test_video_names = test_dataset.readlines()
        self.lasot_home = lasot_home

    def get_next_batch(self, video_name):
        video_name = video_name.strip()
        object_name, number = video_name.split("-")
        path_to_video = os.path.join(self.lasot_home, object_name, video_name)
        with open(os.path.join(path_to_video, "groundtruth.txt")) as gt_boxes_file:
            all_gt_boxes = gt_boxes_file.readlines()

        with open(os.path.join(path_to_video, "nlp.txt")) as gt_nlp_file:
            phrases = gt_nlp_file.readline().strip()

        length = len(all_gt_boxes)
        images = []
        gt_boxes = []
        for frame_num in range(length):
            gt_box = all_gt_boxes[frame_num]
            gt_box = np.array([int(index) for index in gt_box.split(",")], dtype=np.float32)
            images.append(os.path.join(path_to_video, "img", "%08d.jpg" % (frame_num + 1)))
            gt_boxes.append(gt_box)
        return video_name, phrases, images, np.stack(gt_boxes, axis=0)

    def get_all_init_frames(self):
        first_frames = []
        gt_boxes = []
        phrases = []
        for test_video in self.test_video_names:
            video_name = test_video.strip()
            object_name, number = video_name.split("-")
            path_to_video = os.path.join(self.lasot_home, object_name, video_name)
            first_frames.append(os.path.join(path_to_video, "img", "%08d.jpg" % 1))
            with open(os.path.join(path_to_video, "groundtruth.txt")) as gt_boxes_file:
                gt_box = gt_boxes_file.readline()
            gt_box = np.array([int(index) for index in gt_box.split(",")], dtype=np.float32)
            gt_boxes.append(gt_box)
            with open(os.path.join(path_to_video, "nlp.txt")) as gt_nlp_file:
                phrase = gt_nlp_file.readline().strip().replace(".", "")
            phrases.append(phrase)
        return {"first_frames": first_frames, "gt_boxes": gt_boxes, "phrases": phrases}


class OTBDataLoader:

    def __init__(self, otb_test_dataset_file, otb_home):
        with open(otb_test_dataset_file) as test_dataset:
            self.test_video_names = test_dataset.readlines()
        self.otb_home = otb_home

    def get_all_init_frames(self):
        first_frames = []
        gt_boxes = []
        phrases = []
        for test_video in self.test_video_names:
            video_name = test_video.strip()
            path_to_video = os.path.join(self.otb_home, "OTB_videos", video_name)
            first_frames.append(os.path.join(path_to_video, "img", "%08d.jpg" % 1))
            with open(os.path.join(path_to_video, "groundtruth.txt")) as gt_boxes_file:
                gt_box = gt_boxes_file.readline()
            gt_box = np.array([int(index) for index in gt_box.split(",")], dtype=np.float32)
            gt_boxes.append(gt_box)
            with open(os.path.join(self.otb_home, "OTB_queries", video_name + ".txt")) as gt_nlp_file:
                phrase = gt_nlp_file.readline()
            phrases.append(phrase)
        return {"first_frames": first_frames, "gt_boxes": gt_boxes, "phrases": phrases}

    def get_next_batch(self, video_name):
        video_name = video_name.strip()
        path_to_video = os.path.join(self.otb_home, "OTB_videos", video_name)
        with open(os.path.join(path_to_video, "groundtruth.txt")) as gt_boxes_file:
            all_gt_boxes = gt_boxes_file.readlines()
        with open(os.path.join(self.otb_home, "OTB_queries", video_name + ".txt")) as gt_nlp_file:
            phrases = gt_nlp_file.readline()
        length = len(all_gt_boxes)
        images = []
        gt_boxes = []
        for frame_num in range(length):
            image = os.path.join(path_to_video, "img", "%08d.jpg" % (frame_num + 1))
            gt_box = all_gt_boxes[frame_num]
            gt_box = np.array([int(index) for index in gt_box.split(",")], dtype=np.float32)
            images.append(image)
            gt_boxes.append(gt_box)
        return video_name, phrases, images, np.stack(gt_boxes, axis=0)
