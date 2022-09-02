from acquire_video_feature import video_feature
import numpy as np

from multiprocessing import Pool


if __name__ == '__main__':
    index_st = 0
    index_end = 12430

    split_dot = np.linspace(0, 9335, 40)
    split_dot = np.floor(split_dot).astype(np.int64)

    index_infos = []
    for i in range(len(split_dot)-1):
        index_infos.append([split_dot[i], split_dot[i+1]])

    p = Pool(processes=5)
    for index_info in index_infos:
        p.apply_async(video_feature, args=tuple(index_info))
    p.close()
    p.join()




