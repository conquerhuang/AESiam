import os
import torch
import numpy as np
import scipy.io as scio
from tqdm import tqdm

from multiprocessing import Pool

from siamfc import TrackerEvgg_oc

from .datasets_sequences import MultipleDatasets_sequences


def video_feature(index_st, index_end):
    datasets = [
        # 'ILSVRC15',
        'GOT10K',
        # 'LASOT',
        # 'TrackingNet'
    ]
    dataset_paths = {
        'ILSVRC15': r'F:\Cropped_ILSVRC2015',
        'GOT10K': r'F:\Cropped_GOT10K',
        'LASOT': r'F:\Cropped_LaSOT',
        'TrackingNet': r'F:\Cropped_TrackingNet'
    }
    pair_per_seqs = {
        'ILSVRC15': 1,
        'GOT10K': 1,
        'LASOT': 1,
        'TrackingNet': 1
    }

    # 存储视频特征的路径
    video_feature_dir = r'F:\MDESiam\siamvgg\dataset_feature_oc_24'

    # 构建预训练好的跟踪器
    net_path = r'../pretrained/SiamFC/transformed_model_siamvgg_oc24.pth'
    tracker = TrackerEvgg_oc(net_path=net_path)
    tracker.net.eval()

    video_sequences = MultipleDatasets_sequences(datasets=datasets, dataset_paths=dataset_paths,
                                                 pair_per_seqs=pair_per_seqs)
    # 提取所有视频序列的特征
    for i in tqdm(range(index_st, index_end)):
        sequence = video_sequences[i]
        if sequence is None:
            continue
        else:
            root_dir, frames = sequence
            sequence_feature = np.zeros([24, 5, 5], np.float32)
            for frame in frames:
                frame = frame[63:190, 63:190, :]
                frame = np.expand_dims(frame, axis=0)
                frame = frame.astype(np.float32).transpose([0, 3, 1, 2])
                frame = torch.from_numpy(frame).to('cuda:0')
                feature = tracker.net.backbone(frame)
                feature = feature.data.cpu().numpy().squeeze(axis=0)
                sequence_feature = sequence_feature + feature * (1. / len(frames))

            scio.savemat(os.path.join(video_feature_dir, 'seq_' + str(i) + '.mat'),
                         {'feature': sequence_feature, 'root_dir': root_dir})











