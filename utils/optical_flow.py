# COPYRIGHT 2021. Fred Fung. Boston University.
import gzip
import os
import pickle
import re
import shutil
import subprocess
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from absl import flags, logging
from tqdm import tqdm

TAG_FLOAT = 202021.25


class OpticalFlowForVideo:
    def __init__(self, dataset, video_name, lasot_flows, otb_flows):
        self.video_name = video_name
        self.lasot_flows = lasot_flows
        if dataset == 'otb':
            path_to_flows = os.path.join(otb_flows, video_name + '.pklz')
        else:
            object_name = video_name.split('-')[0]
            self.index = 0
            path_to_flows = os.path.join(lasot_flows, object_name,
                                         video_name + '-' + str(self.index) + '.pklz')
        with gzip.open(path_to_flows, 'rb') as f:
            self.flows = pickle.load(f)

    def get_optical_flow(self, frame_id, bounding_box):
        if '%08d' % frame_id in self.flows:
            flow = self.flows['%08d' % frame_id]
        else:
            object_name = self.video_name.split('-')[0]
            self.index += 1
            path_to_flows = os.path.join(self.lasot_flows, object_name,
                                         self.video_name + '-' + str(self.index) + '.pklz')
            with gzip.open(path_to_flows, 'rb') as f:
                self.flows = pickle.load(f)
            if '%08d' % frame_id in self.flows:
                flow = self.flows['%08d' % frame_id]
            else:
                logging.warning('FLOW NOT FOUND: ' + '%08d' % frame_id)
                return np.array([0., 0.], dtype='float32')
        flow_crop = flow[max(0, int(bounding_box[1])): min(flow.shape[0], int(bounding_box[1] + bounding_box[3])),
                    max(0, int(bounding_box[0])): min(flow.shape[1], int(bounding_box[0] + bounding_box[2])), :]
        average_flow = np.mean(flow_crop, axis=(0, 1)).astype('float32')
        if np.isnan(average_flow).any():
            average_flow = np.array([0., 0.], dtype='float32')
        return average_flow


def generate_flownet_inputs_for_lasot():
    with open('LaSOT_testing_set') as test_dataset:
        test_video_names = test_dataset.readlines()
    for video_name in tqdm(test_video_names):
        tqdm.write(video_name)
        video_name = video_name.strip()
        object_name, number = video_name.split('-')
        path_to_video = os.path.join('LaSOTBenchmark', object_name, video_name)
        path_to_frames = path_to_video + '/img/'
        files = generate_flownet_inputs(path_to_frames, output_append='/' + object_name + '/' + video_name + '/')
        list_of_first_frame = files[0]
        list_of_second_frame = files[1]
        list_of_outputs = files[2]
        os.makedirs('/scratch/tmp/' + video_name, exist_ok=True)
        os.makedirs('/scratch/flows/output/' + object_name + '/' + video_name, exist_ok=True)

        with open('/scratch/tmp/' + video_name + '/first_frames_lasot.txt', 'w+') as first_frame_output_file:
            first_frame_output_file.writelines(list_of_first_frame)
        with open('/scratch/tmp/' + video_name + '/second_frames_lasot.txt', 'w+') as second_frame_output_file:
            second_frame_output_file.writelines(list_of_second_frame)
        with open('/scratch/tmp/' + video_name + '/outputs_lasot.txt', 'w+') as outputs_file:
            outputs_file.writelines(list_of_outputs)
        command = 'sudo /scratch/repository/flownet2-docker/run-network.sh -n FlowNet2-s -g 0 -vv ' \
                  '/scratch/tmp/' + video_name + '/first_frames_lasot.txt /scratch/tmp/' + video_name + \
                  '/second_frames_lasot.txt /scratch/tmp/' + video_name + '/outputs_lasot.txt'
        tqdm.write(command)
        subprocess.call([command], shell=True)
        save_flow_as_npy('/scratch/flows/output/' + object_name + '/' + video_name)


def generate_flownet_inputs_for_otb():
    with open('otb_testing_set') as test_dataset:
        test_video_names = test_dataset.readlines()
    for video_name in tqdm(test_video_names):
        tqdm.write(video_name)
        video_name = video_name.strip()
        path_to_frames = '/data/otb_sentences/OTB_videos/' + video_name + '/img/'
        files = generate_flownet_inputs(path_to_frames, output_append='/' + video_name + '/')
        list_of_first_frame = files[0]
        list_of_second_frame = files[1]
        list_of_outputs = files[2]
        with open('/scratch/tmp/' + video_name + 'first_frames_otb.txt', 'w+') as first_frame_output_file:
            first_frame_output_file.writelines(list_of_first_frame)
        with open('/scratch/tmp/' + video_name + 'second_frames_otb.txt', 'w+') as second_frame_output_file:
            second_frame_output_file.writelines(list_of_second_frame)
        with open('/scratch/tmp/' + video_name + 'outputs_otb.txt', 'w+') as outputs_file:
            outputs_file.writelines(list_of_outputs)
        command = 'sudo /scratch/repository/flownet2-docker/run-network.sh -n FlowNet2-s -g 0 -vv ' \
                  '/scratch/tmp/' + video_name + 'first_frames_otb.txt /scratch/tmp/' + video_name + \
                  'second_frames_otb.txt /scratch/tmp/' + video_name + 'outputs_otb.txt'
        tqdm.write(command)
        subprocess.call([command], shell=True)
        save_flow_as_npy('/scratch/flows/output/' + video_name)


def save_flow_as_npy(path_to_flows):
    flow_files = [os.path.join(path_to_flows, f) for f in listdir(path_to_flows) if
                  isfile(join(path_to_flows, f)) and f.endswith('flo')]
    for i in range(0, int(len(flow_files) / 1000) + 1):
        flows = {}
        for flow_file in flow_files[i * 1000: min((i + 1) * 1000, len(flow_files))]:
            flow = read_flow(flow_file)
            frame_id = re.findall(r'\d+', flow_file)[-1]
            flows[frame_id] = flow.astype(np.int32)
        flows_pkl = path_to_flows + '-' + str(i) + '.pklz'
        with gzip.open(flows_pkl, "wb") as pklz:
            pickle.dump(flows, pklz)
    shutil.rmtree(path_to_flows)


def generate_flownet_inputs(path_to_frames, output_append='/'):
    frame_files = [f for f in listdir(path_to_frames) if isfile(join(path_to_frames, f))]
    frame_files.sort()
    list_of_first_frame = []
    list_of_second_frame = []
    list_of_outputs = []
    for i in range(len(frame_files) - 1):
        list_of_first_frame.append(join(path_to_frames, frame_files[i] + '\n'))
        list_of_second_frame.append(join(path_to_frames, frame_files[i + 1] + '\n'))
        list_of_outputs.append('output' + output_append + frame_files[i + 1] + '.flo\n')
    return list_of_first_frame, list_of_second_frame, list_of_outputs


def get_optical_flow_for_target(path_to_frame, bounding_box):
    path_to_flow = os.path.join('/data/lasot_flow/', path_to_frame.split('/')[-4],
                                path_to_frame.split('/')[-3], path_to_frame.split('/')[-1] + '.flo')
    if not os.path.isfile(path_to_flow):
        return [0., 0.]
    flow = read_flow(path_to_flow)
    flow_crop = flow[int(bounding_box[1]): int(bounding_box[1] + bounding_box[3]),
                int(bounding_box[0]): int(bounding_box[0] + bounding_box[2]), :]
    average_flow = np.mean(flow_crop, axis=(0, 1))
    return average_flow


def read_flow(file):
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    if flo_number != TAG_FLOAT:
        logging.warning(file + 'Flow number %r incorrect. Invalid .flo file' % flo_number)
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    # if error try:
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    generate_flownet_inputs_for_lasot()
