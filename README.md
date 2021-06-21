# Siamese Natural Language Tracker: Tracking by Natural Language Descriptions with Siamese Trackers

## Abstract

We propose a novel Siamese Natural Language Tracker (SNLT), which brings the
advancements in visual tracking to the tracking by natural language (NL)
descriptions task. The proposed SNLT is applicable to a wide range of Siamese
trackers, providing a new class of baselines for the tracking by NL task and
promising future improvements from the advancements of Siamese trackers. The
carefully designed architecture of the Siamese Natural Language Region Proposal
Network (SNL-RPN), together with the Dynamic Aggregation of vision and language
modalities, is introduced to perform the tracking by NL task. Empirical results
over tracking benchmarks with NL annotations show that the proposed SNLT
improves Siamese trackers by 3 to 7 percentage points with a slight tradeoff of
speed. The proposed SNLT outperforms all NL trackers to-date and is competitive
among state-of-the-art real-time trackers on LaSOT benchmarks while running at
50 frames per second on a single GPU.

Link to [ArXiv](https://arxiv.org/abs/1912.02048).

```
@inproceedings{feng2021siamese,
title={Siamese Natural Language Tracker: Tracking by Natural Language Descriptions with Siamese Trackers},
author={Feng, Qi and Ablavsky, Vitaly and Bai, Qinxun and Sclaroff, Stan},
booktitle={Proc.\ IEEE Conf.\ on Computer Vision and Pattern Recognition (CVPR)},
pages={},
year={2021}
}
```

## Training:
```bash
# on main machine:
python run_experiment.py --num_machines=$NUM_MACHINES --config_file=$CONFIG --master_ip=$HOSTNAME  --local_machine=0 --num_gpus=$NUM_GPU --master_port=$PORT
# on worker machine:
python run_experiment.py --num_machines=$NUM_MACHINES --config_file=$CONFIG --master_ip=$MASTER_ADDR  --local_machine=$NODE_ID --num_gpus=$NUM_GPU --master_port=$PORT
```

## Testing:
Testing only works on 1 GPU per video, you can run multiple videos on 1 GPU or multiple GPUs.
```bash
# same as training.
python run_experiment.py --num_machines=$NUM_MACHINES --config_file=$CONFIG --master_ip=$HOSTNAME  --local_machine=0 --num_gpus=$NUM_GPU --master_port=$PORT
```
