# COPYRIGHT 2021. Fred Fung. Boston University.
r"""FLAGS/CONFIGS FOR TRAINING AND EVALUATIONS."""

from yacs.config import CfgNode as CN

from input_pipeline.config import get_default_configs as get_default_dataset_configs

_C = CN()
_C.EXPR_NAME = "EXPR"
_C.MODEL = "siamrpn++"
_C.TYPE = "TRAIN"

_C.OUTPUT_SIZE = 25
_C.EXEMPLAR_SIZE = 127
_C.SEARCH_SIZE = 255
_C.LOG_DIR = "/experiments/"

# ------------------------------------------------------------------------ #
# Language network options
# ------------------------------------------------------------------------ #
_C.LANG = CN()
_C.LANG.MODEL = "bert"
_C.LANG.HGLMM_CHECKPOINT = "/checkpoints/hglmm_300.pkl"
_C.LANG.GLOVE_CHECKPOINT = "/checkpoints/glove/glove.42B.300d.pkl"
_C.LANG.BERT_CHECKPOINT = "/checkpoints/bert-base-uncased/"
_C.LANG.FINETUNE_BERT = False

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
_C.TRAIN = CN()
_C.TRAIN.USE_TRIPLET_LOSS = False
_C.TRAIN.EPOCH = 20
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.PRINT_FREQ = 250
_C.TRAIN.BACKBONE_TRAIN = False
_C.TRAIN.BACKBONE_TRAIN_LAYERS = ["layer2", "layer3", "layer4"]
_C.TRAIN.INIT = CN()
_C.TRAIN.INIT.RESUME = ""
_C.TRAIN.INIT.PRETRAINED = ""
_C.TRAIN.INIT.PRETRAINED_RESNET_ONLY = ""
_C.TRAIN.INIT.START_EPOCH = 0
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.LOSS_CLIP_VALUE = 10.

_C.TRAIN.LR = CN()
_C.TRAIN.LR.MOMENTUM = 0.9
_C.TRAIN.LR.WEIGHT_DECAY = 0.5
_C.TRAIN.LR.CLS_WEIGHT = 1.0
_C.TRAIN.LR.LOC_WEIGHT = 1.2
_C.TRAIN.LR.BASE_LR = 0.005
_C.TRAIN.LR.BACKBONE_LAYERS_LR = 0.0005

# For SiamFC
_C.TRAIN.RADIUS = 2

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
_C.DATASET = get_default_dataset_configs()

# ------------------------------------------------------------------------ #
# Anchors options
# ------------------------------------------------------------------------ #
_C.ANCHOR = CN()
_C.ANCHOR.STRIDE = 8
_C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]
_C.ANCHOR.SCALES = [4, 8, 16]
_C.ANCHOR.ANCHOR_NUM = 15
if _C.ANCHOR.ANCHOR_NUM != len(_C.ANCHOR.RATIOS) * len(_C.ANCHOR.SCALES):
    raise ArithmeticError("MISMATCH CONFIGURATION")
_C.ANCHOR.THR_HIGH = 0.6
_C.ANCHOR.THR_LOW = 0.3
_C.ANCHOR.NEG_NUM = 16
_C.ANCHOR.POS_NUM = 16
_C.ANCHOR.TOTAL_NUM = 64

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
_C.TRACK = CN()

# https://drive.google.com/file/d/1yrGwJU8rBC-EOMgPcXZov8fOyt-78KmU/view?usp=sharing
_C.TRACK.RESTORE_FROM = ""
_C.TRACK.DATASET = "otb"
_C.TRACK.USE_OPTICAL_FLOW = False
_C.TRACK.PENALTY_K = 0.04
_C.TRACK.WINDOW_INFLUENCE = 0.44
_C.TRACK.LR = 0.4
_C.TRACK.BASE_SIZE = 8
_C.TRACK.CONTEXT_AMOUNT = 0.5
_C.TRACK.SCORE_SIZE = 25
_C.TRACK.LOST_SCORE_SIZE = 97

_C.TRACK.NLRPN_RATIO = 0.5
_C.TRACK.NLRPN_ALPHA = 1.0 / 300.0

# FOR MMM
_C.TRACK.NUM_FRAMES_HISTORY = 25
_C.TRACK.HISTORY_THRESHOLD = 0.998
_C.TRACK.CONFIDENCE_LOW = 0.0
_C.TRACK.CONFIDENCE_HIGH = 1.0
_C.TRACK.LOST_INSTANCE_SIZE = 831
_C.TRACK.LOST_WINDOW_INFLUENCE = 0.01
_C.TRACK.LOST_LR = 0.8

# FOR SIAMFC ONLY
_C.TRACK.SCALE_STEP = 1.0375
_C.TRACK.NUM_SCALE = 3
_C.TRACK.RESPONSE_UP_STRIDE = 16
_C.TRACK.RESPONSE_SIZE = 17
_C.TRACK.RESPONSE_SCALE = 1E-3
_C.TRACK.GRAY_RATIO = 0.25
_C.TRACK.BLUR_RATIO = 0.15


def get_default_configs():
    return _C.clone()


def get_default_anchor_configs():
    return _C.ANCHOR.clone()
