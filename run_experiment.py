# COPYRIGHT 2021. Fred Fung. Boston University.

import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel

from configs import get_default_configs
from input_pipeline.evaluation import LaSOTDataLoader, OTBDataLoader, TNL2KDataLoader
from input_pipeline.paired_patches_dataset import PairedPatchesDataset, TestDataset
from tracker.inference import Inference
from tracker.siamrpnpp_multinlrpn import SiamRPNPPMultiNLRPN
from tracker.trainer import TrackerTrainer
from utils.logging import get_logger

flags.DEFINE_integer("num_machines", 1, "Number of machines.")
flags.DEFINE_integer("local_machine", 0,
                     "Master node is 0, worker nodes starts from 1. Max should be num_machines - 1.")

flags.DEFINE_integer("num_gpus", 6, "Number of GPUs per machines.")
flags.DEFINE_string("config_file", "experiments/test.yaml", "Default Configuration File.")

flags.DEFINE_string("master_ip", "127.0.0.1", "Master node IP for initialization.")
flags.DEFINE_integer("master_port", 12000, "Master node port for initialization.")

FLAGS = flags.FLAGS


def run_training(rank, cfg):
    dist_rank = rank + cfg.LOCAL_MACHINE * cfg.NUM_GPU_PER_MACHINE
    dist.init_process_group(backend="nccl", rank=dist_rank, world_size=cfg.WORLD_SIZE, init_method=cfg.INIT_METHOD)
    torch.cuda.set_device(rank)
    if cfg.MODEL == "siamrpn++":
        tracker = SiamRPNPPMultiNLRPN(cfg)
    else:
        raise NotImplementedError()
    tracker = tracker.cuda()
    tracker = DistributedDataParallel(tracker, device_ids=[rank], output_device=rank,
                                      broadcast_buffers=cfg.WORLD_SIZE > 1)
    tracker.train()
    train_dataset = PairedPatchesDataset(cfg.DATASET, cfg.ANCHOR, cfg.SEARCH_SIZE, cfg.EXEMPLAR_SIZE, cfg.OUTPUT_SIZE)
    test_dataset = TestDataset(cfg.DATASET, cfg.ANCHOR, cfg.SEARCH_SIZE, cfg.EXEMPLAR_SIZE, cfg.OUTPUT_SIZE)
    trainer = TrackerTrainer(cfg)
    trainer.train_tracker_on_datasets(tracker, train_dataset, test_dataset, dist_rank)
    dist.destroy_process_group()


def run_evaluation(rank, cfg):
    if cfg.MODEL == "siamrpn++":
        tracker = SiamRPNPPMultiNLRPN(cfg)
    else:
        raise NotImplementedError()

    if cfg.TRACK.DATASET == "lasot":
        data_loader = LaSOTDataLoader(cfg.DATASET.LASOT_TEST_FILE, cfg.DATASET.LASOT_TEST_HOME)
    elif cfg.TRACK.DATASET == "otb":
        data_loader = OTBDataLoader(cfg.DATASET.OTB_TEST_FILE, cfg.DATASET.OTB_TEST_HOME)
    elif cfg.TRACK.DATASET == "tnl2k":
        data_loader = TNL2KDataLoader(cfg.DATASET.TNL2K_TEST_HOME)
    else:
        raise NotImplementedError()
    torch.cuda.set_device(rank)
    evaluator = Inference(cfg)
    tracker = evaluator.prepare_tracker(tracker, cfg.TRACK.RESTORE_FROM, rank)
    tracker = tracker.eval()
    tracker = tracker.cuda()
    dir = os.path.join(cfg.LOG_DIR, cfg.TRACK.DATASET, cfg.EXPR_NAME)
    os.makedirs(dir, exist_ok=True)
    _logger = get_logger("main")
    for idx, video_name in enumerate(data_loader.test_video_names):
        if idx % cfg.NUM_GPU_PER_MACHINE == rank:
            try:
                _logger.info("Evaluate %s on GPU %d." % (video_name.strip(), rank))
                video_name, phrase, video, gt_boxes = data_loader.get_next_batch(video_name=video_name)
                results = evaluator.track_on_video(tracker, video, gt_boxes, phrase)
                with open(os.path.join(dir, video_name.strip() + ".txt"), "w") as f:
                    f.writelines(results["boxes"])
                _logger.info("Mean IOU on %s: %.4f" % (video_name, np.mean(results["ious"])))
                _logger.info("NL Correlation is \t %.2f" % np.corrcoef(results["ious"][1:], results["nl_ents"])[0, 1])
                _logger.info("VIS Correlation is \t %.2f" % np.corrcoef(results["ious"][1:], results["ents"])[0, 1])
            except Exception as ex:
                _logger.warning("Failed: %s" % video_name)


if __name__ == "__main__":
    FLAGS(sys.argv)
    cfg = get_default_configs()
    cfg.merge_from_file(FLAGS.config_file)
    cfg.NUM_GPU_PER_MACHINE = FLAGS.num_gpus
    cfg.NUM_MACHINES = FLAGS.num_machines
    cfg.LOCAL_MACHINE = FLAGS.local_machine
    cfg.WORLD_SIZE = FLAGS.num_machines * FLAGS.num_gpus
    cfg.EXPR_NAME = cfg.EXPR_NAME + "_" + datetime.now().strftime("%m_%d.%H:%M:%S.%f")
    if cfg.TYPE == "TRAIN":
        cfg.INIT_METHOD = "tcp://%s:%d" % (FLAGS.master_ip, FLAGS.master_port)
        mp.spawn(run_training, args=(cfg,), nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
    elif cfg.TYPE == "INFERENCE":
        mp.spawn(run_evaluation, args=(cfg,), nprocs=cfg.NUM_GPU_PER_MACHINE, join=True)
