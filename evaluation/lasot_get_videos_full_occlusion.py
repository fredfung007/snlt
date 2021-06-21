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

import numpy as np


def getMaxLength(arr, n):
    # intitialize count
    count = 0
    # initialize max
    result = 0
    time = 0
    for i in range(0, n):
        if (arr[i] == 0):
            count = 0
        else:
            count += 1
            if result < count:
                result = count
                time = i
    return result, time


base_dir = '/research/fung/data/LaSOTBenchmark'
tracker_base_dir = '/research/fung/tracker-eval'
testing_set = '/research/fung/client1/research/tell_me_what_to_track/tracking/data/LaSOT_testing_set'

with open(testing_set) as f:
    testing_videos = f.readlines()

mask = []
videos = []
times = []
for video in testing_videos:
    video = video.strip()
    object, id = video.split('-')
    occlusion_file = os.path.join(base_dir, object, video, 'full_occlusion.txt')
    out_of_view_file = os.path.join(base_dir, object, video, 'out_of_view.txt')
    occlusions = np.loadtxt(occlusion_file, delimiter=',')
    max_consecutive_occlusion, time = getMaxLength(occlusions, len(occlusions))
    num_out_of_view = np.count_nonzero(np.loadtxt(out_of_view_file, delimiter=','))
    mask.append(max_consecutive_occlusion >= 10)
    num_occlusion = np.count_nonzero(occlusions)
    if (num_occlusion == 0) and (num_out_of_view > 10 and num_out_of_view < 50):
        videos.append(video)
        times.append(time)
print(len(videos))
for v in videos:
    print(v)
