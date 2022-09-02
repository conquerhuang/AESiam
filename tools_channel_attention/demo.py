from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerESiamVggAtten


if __name__ == '__main__':
    # setting of demo video sequence
    seq_dir = os.path.expanduser(r'E:\dataset\tracker_evaluate_dataset\OTB/Bird2/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')

    # hyper parameter setting
    net_path = r'../pretrained/SiamFC/ESiamvggAtten.pth'
    scale_num = 3
    scale_step = 1.03275
    scale_lr = 0.54
    scale_penalty = 0.9745
    window_influence = 0.23

    # track init
    tracker = TrackerESiamVggAtten(
        net_path=net_path, name='SiamVggAtten',
        scale_step=scale_step,
        scale_lr=scale_lr,
        scale_penalty=scale_penalty,
        window_influence=window_influence,
        scale_num=scale_num
    )
    tracker.track(img_files, anno[0], visualize=True)
