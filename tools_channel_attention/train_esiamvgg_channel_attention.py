# from __future__ import absolute_import

import os
# from got10k.datasets import *

from siamfc import TrackerESiamVggAtten
from siamfc.datasets import TrackingNetCropped
from siamfc.datasets import Got10kCropped
from siamfc.datasets import ILSVRC2015Cropped
from siamfc.datasets import LaSOTCropped
from siamfc.datasets import MultipleDatasets


if __name__ == '__main__':
    # update on the backbone of ESiamVgg

    net_path = r'./transformed_atten_dict.pth'
    initial_lr = 1e-2
    ultimate_lr = 1e-5
    epoch_num = 50

    datasets = [
        # 'ILSVRC15',
        'GOT10K',
        # 'LASOT',
        # 'TrackingNet'
    ]
    dataset_paths = {
        'ILSVRC15':     r'E:\Cropped_ILSVRC2015',
         'GOT10K':      r'E:\Cropped_GOT10K',
         'LASOT':       r'E:\Cropped_LaSOT',
         'TrackingNet': r'E:\Cropped_TrackingNet'
    }
    pair_per_seqs = {
        # 'ILSVRC15':    5 * epoch_num,
        'GOT10K':      1 * epoch_num,
        # 'LASOT':       1 * epoch_num,
        # 'TrackingNet': 1 * epoch_num,
    }
    train_dataset = MultipleDatasets(datasets=datasets, dataset_paths=dataset_paths, pair_per_seqs=pair_per_seqs)

    tracker = TrackerESiamVggAtten(
        name='ESiamvggAtten',
        net_path=net_path,
        initial_lr=initial_lr,
        ultimate_lr=ultimate_lr,
        epoch_num=epoch_num
    )
    tracker.train_over(train_dataset)