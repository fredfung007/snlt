# COPYRIGHT 2021. Fred Fung. Boston University.
import random

import cv2
import numpy as np
import scipy.special
import torch

from evaluation.ope import overlap_ratio
from utils.anchor import Anchors
from utils.logging import get_logger
from utils import check_keys
from utils.optical_flow import OpticalFlowForVideo
from utils.tracking_inference import TrackingState, get_subwindow, change, convert_bbox, bbox_clip, compute_size


class Inference:
    def __init__(self, inference_cfg):
        self.inf_cfg = inference_cfg
        self.anchor_num = len(self.inf_cfg.ANCHOR.RATIOS) * len(self.inf_cfg.ANCHOR.SCALES)
        self.window = np.tile(
            np.outer(np.hanning(self.inf_cfg.TRACK.SCORE_SIZE), np.hanning(self.inf_cfg.TRACK.SCORE_SIZE))[np.newaxis,
            :, :], [self.anchor_num, 1, 1])
        self.anchors = Anchors(self.inf_cfg.ANCHOR.STRIDE, self.inf_cfg.ANCHOR.RATIOS,
                               self.inf_cfg.ANCHOR.SCALES)
        self.anchors.generate_all_anchors(self.inf_cfg.SEARCH_SIZE // 2, self.inf_cfg.OUTPUT_SIZE)
        self.lost_window = np.tile(
            np.outer(np.hanning(self.inf_cfg.TRACK.LOST_SCORE_SIZE), np.hanning(self.inf_cfg.TRACK.LOST_SCORE_SIZE))[
            np.newaxis, :, :], [self.anchor_num, 1, 1])
        self.lost_anchors = Anchors(self.inf_cfg.ANCHOR.STRIDE, self.inf_cfg.ANCHOR.RATIOS,
                                    self.inf_cfg.ANCHOR.SCALES)
        self.lost_anchors.generate_all_anchors(self.inf_cfg.TRACK.LOST_INSTANCE_SIZE // 2,
                                               self.inf_cfg.TRACK.LOST_SCORE_SIZE)

        self._logger = get_logger(__name__)

    def prepare_tracker(self, tracker, restore_from, rank=0):
        ckpt = torch.load(restore_from, map_location=lambda storage, loc: storage.cpu())
        restore_kv = {key.replace("module.", ""): ckpt["state_dict"][key] for key in ckpt["state_dict"].keys()}
        check_keys(tracker, restore_kv, rank=rank)
        tracker.load_state_dict(restore_kv, strict=True)
        if rank == 0:
            self._logger.info("RESTORED TRACKER FROM {}.".format(restore_from))
        return tracker

    def track_on_video(self, tracker, video, gt_boxes, phrase, debug=False):
        first_frame_path = video[0]
        z_crop = self._initialize_tracker(first_frame_path, gt_boxes[0])
        if debug:
            template = np.transpose(z_crop[0].cpu().numpy(), [1, 2, 0])
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB) / 255.
            import matplotlib.pyplot as pyplot
            pyplot.imshow(template)
            pyplot.axis('off')
            pyplot.show()
            self._logger.info(phrase)
        boxes = []
        ents = []
        nl_ents = []
        for frame_id, frame_path in enumerate(video[1:]):
            frame_id += 1
            frame = cv2.imread(frame_path)
            scale_z, size, x_crop, crop_center = self._crop_and_get_center_pos(frame, frame_id + 1)
            frame_data = {"template": z_crop, "search": x_crop, "phrase": [phrase]}
            with torch.autograd.no_grad():
                frame_results = tracker(frame_data)

            score = frame_results["cls"][0].cpu().numpy()
            nl_score = frame_results["nl_cls"][0].cpu().numpy()
            if self.tracking_state == TrackingState.STABLE:
                pred_bbox = convert_bbox(frame_results["reg"], self.anchors.all_anchors[1])
                nl_pred_bbox = convert_bbox(frame_results["nl_reg"], self.anchors.all_anchors[1])
            elif self.tracking_state == TrackingState.LOST:
                pred_bbox = convert_bbox(frame_results["reg"], self.lost_anchors.all_anchors[1])
                nl_pred_bbox = convert_bbox(frame_results["nl_reg"], self.lost_anchors.all_anchors[1])
            else:
                raise NotImplementedError()
            track_result = self.make_prediction(pred_bbox, score, nl_pred_bbox, nl_score, crop_center, frame.shape[:2],
                                                scale_z, size)
            boxes.append("%.2f,%.2f,%.2f,%.2f\n" %
                         (track_result["bbox"][0], track_result["bbox"][1],
                          track_result["bbox"][2], track_result["bbox"][3]))

            iou = overlap_ratio(gt_boxes[frame_id], track_result["bbox"])[0]
            ents.append(track_result["ent"])
            nl_ents.append(track_result["nl_ent"])
            track_result["crop"] = crop_center
            track_result["ious"] = iou
            self.predictions.append(track_result)
            self.history_management(debug)
        ious = [pred["ious"] for pred in self.predictions]
        return {"boxes": boxes, "ious": ious, "nl_ents": nl_ents, "ents": ents}

    def _initialize_tracker(self, frame_path, xywh_bbox):
        img = cv2.imread(frame_path)
        center_pos = np.array([xywh_bbox[0] + (xywh_bbox[2] - 1) / 2., xywh_bbox[1] + (xywh_bbox[3] - 1) / 2.])
        size = np.array([xywh_bbox[2], xywh_bbox[3]])
        z_crop = self.compute_template(img, xywh_bbox)
        self.channel_average = np.mean(img, axis=(0, 1))
        if self.inf_cfg.TRACK.USE_OPTICAL_FLOW:
            self.optical_flow = OpticalFlowForVideo(dataset=self.inf_cfg.TRACK.DATASET,
                                                    video_name=frame_path.split("/")[-3],
                                                    lasot_flows=self.inf_cfg.DATASET.LASOT_FLOW,
                                                    otb_flows=self.inf_cfg.DATASET.OTB_FLOW)
        self.tracking_state = TrackingState.STABLE
        self.lost_count = 0
        self.predictions = [
            {"center_pos": center_pos, "size": size, "ious": 1.0, "best_score": 1.0, "nl_std": 0.1, "std": 0.1}]
        return z_crop

    def compute_template(self, img, xywh_bbox):
        center_pos = np.array([xywh_bbox[0] + (xywh_bbox[2] - 1) / 2, xywh_bbox[1] + (xywh_bbox[3] - 1) / 2])
        size = np.array([xywh_bbox[2], xywh_bbox[3]])
        w_z = size[0] + self.inf_cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + self.inf_cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = self.inf_cfg.EXEMPLAR_SIZE / s_z
        channel_average = np.mean(img, axis=(0, 1))
        z_crop = get_subwindow(img, center_pos, self.inf_cfg.EXEMPLAR_SIZE, scale_z, channel_average)
        return z_crop

    def _crop_and_get_center_pos(self, img, frame_id):
        if self.tracking_state == TrackingState.STABLE:
            center_pos = self.predictions[-1]["center_pos"]
            size = self.predictions[-1]["size"]
            if self.inf_cfg.TRACK.USE_OPTICAL_FLOW:
                bbox = np.stack([center_pos[0] - size[0] / 2, center_pos[1] - size[1] / 2, size[0], size[1]], axis=0)
                optical_flow = self.optical_flow.get_optical_flow(frame_id, bbox)
                crop_center = center_pos + optical_flow
            else:
                crop_center = center_pos
        elif self.tracking_state == TrackingState.LOST:
            past_prediction = random.sample(self.predictions[int(-1.5 * self.lost_count) - 1:], k=1)[0]
            crop_center = past_prediction["center_pos"]
            size = past_prediction["size"]
        else:
            raise NotImplementedError()
        w_z = size[0] + self.inf_cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + self.inf_cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.inf_cfg.EXEMPLAR_SIZE / s_z
        if self.tracking_state == TrackingState.STABLE:
            x_crop = get_subwindow(img, crop_center, self.inf_cfg.SEARCH_SIZE, scale_z, self.channel_average)
        elif self.tracking_state == TrackingState.LOST:
            x_crop = get_subwindow(img, crop_center, self.inf_cfg.TRACK.LOST_INSTANCE_SIZE, scale_z,
                                   self.channel_average)
        else:
            raise NotImplementedError()
        return scale_z, size, x_crop, crop_center

    def make_prediction(self, vis_bbox, score, nl_pred_bbox, nl_score, crop_center, img_shape, scale, size):
        score = scipy.special.softmax(score, axis=-1)[:, :, :, 1]
        nl_score = scipy.special.softmax(nl_score, axis=-1)[:, :, :, 1]
        std = np.std(score)
        ent = np.sum(scipy.special.entr(score))
        nl_ent = np.sum(scipy.special.entr(nl_score))
        nl_std = np.std(nl_score)
        w_vis, w_nl = scipy.special.softmax(
            [self.inf_cfg.TRACK.NLRPN_ALPHA * nl_ent, self.inf_cfg.TRACK.NLRPN_ALPHA * ent])
        pred_bbox = w_vis * vis_bbox + w_nl * nl_pred_bbox
        pred_score = w_vis * score + w_nl * nl_score
        if self.tracking_state == TrackingState.STABLE:
            width = pred_bbox[2, :]
            height = pred_bbox[3, :]
            s_c = change(compute_size(width, height) / (compute_size(size[0] * scale, size[1] * scale)))
            r_c = change((size[0] / size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
            penalty = np.exp(-(r_c * s_c - 1) * self.inf_cfg.TRACK.PENALTY_K)
            pred_score = penalty * pred_score
            pred_score = (1. - self.inf_cfg.TRACK.WINDOW_INFLUENCE) * pred_score
            pred_score += self.inf_cfg.TRACK.WINDOW_INFLUENCE * self.window
        elif self.tracking_state == TrackingState.LOST:
            pred_score = (1. - self.inf_cfg.TRACK.LOST_WINDOW_INFLUENCE) * pred_score
            pred_score += self.inf_cfg.TRACK.LOST_WINDOW_INFLUENCE * self.lost_window
        else:
            raise AssertionError("TRACKING STATE WRONG " + str(self.tracking_state))

        idx, w, h = np.unravel_index(np.argmax(pred_score), pred_score.shape)
        bbox = pred_bbox[:, idx, w, h] / scale

        if self.tracking_state == TrackingState.STABLE:
            lr = pred_score[idx, w, h] * self.inf_cfg.TRACK.LR
        elif self.tracking_state == TrackingState.LOST:
            lr = pred_score[idx, w, h] * self.inf_cfg.TRACK.LOST_LR
        else:
            raise NotImplementedError()
        width = size[0] * (1 - lr) + bbox[2] * lr
        height = size[1] * (1 - lr) + bbox[3] * lr
        if self.tracking_state == TrackingState.STABLE:
            cx = bbox[0] + crop_center[0] - self.inf_cfg.SEARCH_SIZE // 2 / scale
            cy = bbox[1] + crop_center[1] - self.inf_cfg.SEARCH_SIZE // 2 / scale
        elif self.tracking_state == TrackingState.LOST:
            cx = bbox[0] + crop_center[0] - self.inf_cfg.TRACK.LOST_INSTANCE_SIZE // 2 / scale
            cy = bbox[1] + crop_center[1] - self.inf_cfg.TRACK.LOST_INSTANCE_SIZE // 2 / scale
        else:
            raise NotImplementedError()
        cx, cy, width, height = bbox_clip(cx, cy, width, height, img_shape)
        bbox = np.array([cx - width / 2, cy - height / 2, width, height])
        best_score = pred_score[idx, w, h]
        return {"bbox": bbox, "best_score": best_score, "center_pos": np.array([cx, cy]),
                "size": np.array([width, height]), "std": std, "nl_std": nl_std, "ent": ent, "nl_ent": nl_ent}

    def history_management(self, debug=False):
        past_pred = self.predictions[-5:]
        score = np.mean([pred["best_score"] for pred in past_pred])
        if self.tracking_state == TrackingState.STABLE:
            self.lost_count = 0
            score_status = score < self.inf_cfg.TRACK.CONFIDENCE_LOW
            if score_status:
                self.lost_count += 1
                self.tracking_state = TrackingState.LOST
                if debug:
                    self._logger.warning("LOST MODE.")
                    self._logger.warning("SCORE IS %.2f" % score)
        elif self.tracking_state == TrackingState.LOST:
            self.lost_count += 1
            if debug:
                self._logger.info("LOST_COUNT: %d" % self.lost_count)
            score_status = score > self.inf_cfg.TRACK.CONFIDENCE_HIGH
            if score_status:
                self.tracking_state = TrackingState.STABLE
                if debug:
                    self._logger.warning("STABLE MODE.")
                    self._logger.warning("SCORE IS %.2f" % score)
