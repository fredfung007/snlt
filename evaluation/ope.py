# ------------------------------------------------------------------------------
# CONFIDENTIAL AND PROPRIETARY.
#
# COPYRIGHT (c) 2020. Fred Fung. ALL RIGHTS RESERVED.
#
# Unauthorized use or disclosure in any manner may result in disciplinary
# action up to and including termination of employment (in the case of
# employees), termination of an assignment or contract (in the case of
# contingent staff), and potential civil and criminal liability.
#
# For internal use only.
# ------------------------------------------------------------------------------
import os
import re

import numpy as np
import pandas as pd


def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    if rect1.ndim == 1:
        rect1 = rect1[np.newaxis, :]
    if rect2.ndim == 1:
        rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def success_error(gt_center, result_center, thresholds, n_frame):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    for i in range(len(thresholds)):
        success[i] = np.sum(dist <= thresholds[i]) / float(n_frame)
    return success


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def convert_bb_to_norm_center(bboxes, gt_wh):
    return convert_bb_to_center(bboxes) / (gt_wh + 1e-16)


def get_trajectories(basedir, evaldir, tracker_name, category, video):
    if category is not None:
        gt_file = os.path.join(basedir, category, video, 'groundtruth.txt')
    else:
        gt_file = os.path.join(basedir, video, 'groundtruth.txt')
    with open(gt_file) as f:
        all_gt_boxes_str = f.readlines()
    all_gt_boxes = []
    for gt_box in all_gt_boxes_str:
        gt_box = np.array([float(index) for index in gt_box.split(',')])
        all_gt_boxes.append(gt_box)
    all_gt_boxes = np.array(all_gt_boxes)
    output_file = os.path.join(evaldir, tracker_name, video + '.txt')
    with open(output_file) as f:
        all_output_boxes_str = f.readlines()
    all_output_boxes = []
    for output_box in all_output_boxes_str:
        output_box = np.array([float(index) for index in re.split(',|\t', output_box)])
        all_output_boxes.append(output_box)
    all_output_boxes = np.array(all_output_boxes)
    if video == 'monkey-17':
        all_output_boxes = all_output_boxes[:len(all_gt_boxes)]
    if len(all_gt_boxes) == 1 + len(all_output_boxes):
        all_output_boxes = np.concatenate([all_gt_boxes[0:1], all_output_boxes])
    if len(all_gt_boxes) == 2 + len(all_output_boxes):
        all_output_boxes = np.concatenate([all_gt_boxes[0:1], all_output_boxes])
        all_gt_boxes = all_gt_boxes[:-1]
    if len(all_gt_boxes) == 3 + len(all_output_boxes):
        all_gt_boxes = all_gt_boxes[3:]
    elif len(all_gt_boxes) < len(all_output_boxes):
        all_output_boxes = all_output_boxes[:len(all_gt_boxes)]
    elif len(all_gt_boxes) > len(all_output_boxes):
        raise ValueError('size of prediction and gt mismatch')
    return all_gt_boxes, all_output_boxes


def eval_success(tracker_name, basedir, evaldir, video_names, categories=None):
    """
    Args:
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """
    success_ret = {}
    for i, video in enumerate(video_names):
        all_gt_boxes, all_output_boxes = get_trajectories(basedir, evaldir, tracker_name,
                                                          categories[i] if categories is not None else None, video)
        success_ret[video] = success_overlap(all_gt_boxes, all_output_boxes, len(all_gt_boxes))
    return success_ret


def eval_precision(tracker_name, basedir, evaldir, video_names, categories=None):
    """
    Args:
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """
    precision_ret = {}
    for i, video in enumerate(video_names):
        all_gt_boxes, all_output_boxes = get_trajectories(basedir, evaldir, tracker_name,
                                                          categories[i] if categories is not None else None, video)
        gt_center = convert_bb_to_center(all_gt_boxes)
        tracker_center = convert_bb_to_center(all_output_boxes)
        thresholds = np.arange(0, 51, 1)
        precision_ret[video] = success_error(gt_center, tracker_center, thresholds, len(all_gt_boxes))
    return precision_ret


def eval_norm_precision(tracker_name, basedir, evaldir, video_names, categories=None):
    """
    Args:
        eval_trackers: list of tracker name or single tracker name
    Return:
        res: dict of results
    """

    norm_precision_ret = {}
    for i, video in enumerate(video_names):
        all_gt_boxes, all_output_boxes = get_trajectories(basedir, evaldir, tracker_name,
                                                          categories[i] if categories is not None else None, video)
        gt_center_norm = convert_bb_to_norm_center(all_gt_boxes, all_gt_boxes[:, 2:4])
        tracker_center_norm = convert_bb_to_norm_center(all_output_boxes, all_gt_boxes[:, 2:4])
        thresholds = np.arange(0, 51, 1) / 100
        norm_precision_ret[video] = success_error(gt_center_norm, tracker_center_norm, thresholds, len(all_gt_boxes))
    return norm_precision_ret


def success_overlap_iou(ious):
    n_frame = len(ious)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(ious > thresholds_overlap[i]) / float(n_frame)
    return success


def get_label_for_ablation_study(tracker):
    if tracker == 'SIAMRPNPP_BASELINE':
        return 'PP[B]'
    if tracker == 'SIAMRPNPP/HGLMM/FINALIZED':
        return 'PP[H]'
    if tracker == 'SIAMRPNPP/GLOVE/FINALIZED':
        return 'PP[G]'
    if tracker == 'SIAMRPNPP/HGLMM/NL_INIT':
        return 'PP[NL]'
    if tracker == 'SIAMRPN_BASELINE':
        return 'RPN[B]'
    if tracker == 'SIAMRPN/GLOVE/FINALIZED':
        return 'RPN[G]'
    if tracker == 'SIAMRPN/HGLMM/FINALIZED':
        return 'RPN[H]'
    if tracker == 'SiamFC_BASELINE':
        return 'FC[B]'
    if tracker == "dimp50_000":
        return 'DiMP'
    if tracker == "prdimp50_000":
        return 'PrDiMP'
    if tracker == "V3-BEST":
        return 'PP[BERT]'
    if tracker == "siamrcnn_lasot":
        return 'SiamRCNN'
    if tracker == "FENG_NL":
        return "FENG[NL]"
    return tracker


def eval_lasot():
    base_dir = '/research/fung/data/LaSOTBenchmark'
    testing_set = '/research/fung/repository/lang_tracking_vos/evaluation/nl-consistent-lasot.txt'
    tracker_base_dir = '/research/fung/results/cvpr_results/lasot'
    trackers = [
        "dimp50_000",
        "prdimp50_000",
    ]

    with open(testing_set, 'r') as f:
        testing_videos = f.readlines()
    categories = [test_video.split('-')[0] for test_video in testing_videos]
    testing_videos = [test_video.strip() for test_video in testing_videos]

    thresholds = np.arange(0, 1.05, 0.05)
    success_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        success_ret = eval_success(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        success_df[get_label_for_ablation_study(tracker)] = np.mean(list(success_ret.values()), axis=0)
        auc = np.mean(list(success_ret.values()))
        print(auc)
    print(success_df.to_string(index=False))

    # # Evaluation precision plot.
    thresholds = np.arange(0, 51, 1)
    precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_precision(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        print(precision)
    print(precision_df.to_string(index=False))

    # Normalized Evaluation precision plot.
    thresholds = np.arange(0, 51, 1) / 100.
    normed_precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_norm_precision(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        normed_precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        print(precision)
    print(normed_precision_df.to_string(index=False))


def eval_nl_lasot():
    base_dir = '/research/fung/data/LaSOTBenchmark'
    tracker_base_dir = '/research/fung/eccv_results/lasot'
    testing_set = 'experiments/eccv/finalized/NL-CONSISTENT-LASOT'
    trackers = [
        # 'VITAL', 'MDNet', 'ATOM', 'ECO','MEEM',
        'SIAMRPNPP_BASELINE', 'SIAMRPNPP/HGLMM/FINALIZED',
        'SIAMRPNPP/GLOVE/FINALIZED',
        # 'SIAMRPNPP/HGLMM/NL_INIT',
        # 'SiamFC_BASELINE',
        'SIAMRPN_BASELINE', 'SIAMRPN/HGLMM/FINALIZED',
        'SIAMRPN/GLOVE/FINALIZED'
    ]

    with open(testing_set, 'r') as f:
        testing_videos = f.readlines()
    categories = [test_video.split('-')[0] for test_video in testing_videos]
    testing_videos = [test_video.strip() for test_video in testing_videos]

    thresholds = np.arange(0, 1.05, 0.05)
    success_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        success_ret = eval_success(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        success_df[get_label_for_ablation_study(tracker)] = np.mean(list(success_ret.values()), axis=0)
        auc = np.mean(list(success_ret.values()))
        print(auc)
    print(success_df.to_string(index=False))

    # Evaluation precision plot.
    thresholds = np.arange(0, 51, 1)
    precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_precision(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        print(precision)
    print(precision_df.to_string(index=False))

    # Normalized Evaluation precision plot.
    thresholds = np.arange(0, 51, 1) / 100.
    normed_precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_norm_precision(tracker, base_dir, tracker_base_dir, testing_videos, categories)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        normed_precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        print(precision)
    print(normed_precision_df.to_string(index=False))


def eval_otb():
    base_dir = '/research/fung/data/otb_sentences/OTB_videos'
    tracker_base_dir = '/research/fung/cvpr_results/lasot'
    testing_set = '/research/fung/client1/research/tmwtt_v2/input_pipeline/otb_testing_set'
    trackers = [
        "dimp18_000",
        "prdimp18_000",
    ]

    with open(testing_set, 'r') as f:
        testing_videos = f.readlines()
    testing_videos = [test_video.strip() for test_video in testing_videos]

    thresholds = np.arange(0, 1.05, 0.05)
    success_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        success_ret = eval_success(tracker, base_dir, tracker_base_dir, testing_videos)
        success_df[get_label_for_ablation_study(tracker)] = np.mean(list(success_ret.values()), axis=0)
        auc = np.mean(list(success_ret.values()))
        print(auc)
    print(success_df.to_string(index=False))

    # # Evaluation precision plot.
    thresholds = np.arange(0, 51, 1)
    precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_precision(tracker, base_dir, tracker_base_dir, testing_videos)
        precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        print(precision)
    print(precision_df.to_string(index=False))

    # Normalized Evaluation precision plot.
    thresholds = np.arange(0, 51, 1) / 100.
    normed_precision_df = pd.DataFrame(thresholds, columns=['THRESHOLD'])

    for idx, tracker in enumerate(trackers):
        precision_ret = eval_norm_precision(tracker, base_dir, tracker_base_dir, testing_videos)
        precision = np.mean(list(precision_ret.values()), axis=0)[20]
        normed_precision_df[get_label_for_ablation_study(tracker)] = np.mean(list(precision_ret.values()), axis=0)
        print(precision)
    print(normed_precision_df.to_string(index=False))


def lasot_per_video_iou():
    base_dir = '/research/fung/data/LaSOTBenchmark'
    tracker_base_dir = '/research/fung/cvpr_results/lasot'
    testing_set = '/research/fung/client1/research/tmwtt_v2/input_pipeline/LaSOT_testing_set'
    # tracker = "prdimp18_000"
    tracker = "V3-BEST"

    with open(testing_set, 'r') as f:
        testing_videos = f.readlines()
    categories = [test_video.split('-')[0] for test_video in testing_videos]
    testing_videos = [test_video.strip() for test_video in testing_videos]
    for i, video in enumerate(testing_videos):
        all_gt_boxes, all_output_boxes = get_trajectories(base_dir, tracker_base_dir, tracker,
                                                          categories[i] if categories is not None else None, video)
        iou = overlap_ratio(all_gt_boxes, all_output_boxes)
        # success = success_overlap_iou(iou)
        print('%s\t%.2f' % (video, np.mean(iou)))


def lasot_per_cat_success():
    base_dir = '/research/fung/data/LaSOTBenchmark'
    testing_set = '/research/fung/client1/research/tmwtt_v2/input_pipeline/LaSOT_testing_set'
    tracker_base_dir = '/research/fung/cvpr_results/lasot'
    trackers = [
        "prdimp18_000",
        "V3-BEST"
    ]
    # tracker_base_dir = '/research/fung/cvpr_results/lasot'
    # trackers = ['SiamRPN++_tracking_result']
    with open(testing_set, 'r') as f:
        testing_videos = f.readlines()
    categories = [test_video.split('-')[0] for test_video in testing_videos]
    testing_videos = [test_video.strip() for test_video in testing_videos]
    ious = []
    per_cat = {"tracker": []}
    for c in categories:
        per_cat[c] = []

    for tracker in trackers:
        per_cat["tracker"].append(tracker)
        for i, video in enumerate(testing_videos):
            if i % 4 == 0:
                ious = []
            all_gt_boxes, all_output_boxes = get_trajectories(base_dir, tracker_base_dir, tracker,
                                                              categories[i] if categories is not None else None, video)
            iou = overlap_ratio(all_gt_boxes, all_output_boxes)
            ious = ious + list(iou)
            if i % 4 == 3:
                cat_success = success_overlap_iou(ious)
                per_cat[video.split('-')[0]].append("%.2f" % np.mean(cat_success))
    for key in per_cat:
        row = str(key)
        for elem in per_cat[key]:
            row = row + "\t" + elem
        print(row)


if __name__ == '__main__':
    eval_lasot()
