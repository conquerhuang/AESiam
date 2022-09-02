from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerESiamVgg


if __name__ == '__main__':
    seq_dir = os.path.expanduser(r'E:\dataset\tracker_evaluate_dataset\OTB/Bird2/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = r'transformed_esiamvgg.pth'
    tracker = TrackerESiamVgg(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
