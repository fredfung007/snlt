# COPYRIGHT 2021. Fred Fung. Boston University.
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
# ["vg", "mscoco", "lasot", "youtube", "davis", "otb", "got"]
_C.DATASETS = ["lasot"]
_C.TEMPLATE = CN()
_C.TEMPLATE.SHIFT = 4
_C.TEMPLATE.SCALE = 0.05
_C.TEMPLATE.BLUR = 0.0
_C.TEMPLATE.FLIP = 0.0
_C.TEMPLATE.COLOR = 1.0
_C.SEARCH = CN()
_C.SEARCH.SHIFT = 64
_C.SEARCH.SCALE = 0.18
_C.SEARCH.BLUR = 0.0
_C.SEARCH.FLIP = 0.0
_C.SEARCH.COLOR = 1.0
_C.NEG = 0.2
_C.GRAY = 0.0

# TODO: Path to all datasets
_C.DATA_HOME = ""

_C.VISUAL_GENOME_HOME = "/vg"
_C.VISUAL_GENOME_JSON_FILE = "/vg/crop.json"
_C.MSCOCO_JSON_FILE = "/mscoco/crop511/train2017.json"
_C.MSCOCO_HOME = "/mscoco/"
_C.LASOT_JSON_FILE = "/lasotcrops/lasot.json"
_C.LASOT_TEST_JSON_FILE = "/lasotcrops/lasot-eval.json"
_C.LASOT_CROP_HOME = "/lasotcrops/"
_C.LASOT_NUM_PER_VIDEO = 2500
_C.OTB_JSON_FILE = "/otb_crops/otb_training.json"
_C.OTB_CROP_HOME = "/otb_crops/"
_C.OTB_NUM_PER_VIDEO = 1000
_C.YOUTUBE_JSON_FILE = "/youtube-bb-crop/yt_bb_detection_train.json"
_C.YOUTUBE_HOME = "/youtube-bb-crop/"
_C.YOUTUBE_NUM_PER_VIDEO = 25
_C.GOT_JSON_FILE = "/got-crop/got10k-train.json"
_C.GOT_HOME = "/got-crop/"
_C.GOT_NUM_PER_VIDEO = 25
_C.DAVIS_JSON_FILE = "/DAVIS/crop/2017_train.json"
_C.DAVIS_HOME = "/DAVIS/crop"
_C.ALWAYS_USE_FIRST_FRAME_Z = False
_C.VIDEOS_PER_EPOCH = 600000
_C.VIDEOS_FOR_TESTING = 10240

# TODO: File that contain track names.
_C.LASOT_TEST_FILE = "LaSOT_testing_set"
_C.OTB_TEST_FILE = "otb_testing_set"
_C.LASOT_TEST_HOME = "LaSOTBenchmark/"
_C.OTB_TEST_HOME = "otb_sentences/"
_C.LASOT_FLOW = "lasot_flows"
_C.OTB_FLOW = "otb_flows"
# Tested on a older version. Use with causion.
_C.TNL2K_TEST_HOME = "tnl2k-revised"

def get_default_configs():
    return _C.clone()
