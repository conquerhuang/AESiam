from __future__ import absolute_import, division

import copy

import numpy as np
import torch
import cv2
from glob import glob
import os
import pickle
from torch.utils.data import Dataset


class MDESiamDataset(Dataset):
    """
        MDESiam train dataset for second step. You shall put the cluster result into folder './cluster_result'
        first.
    """
    def __init__(self, dataset_paths, transforms=None, pair_per_seqs=1):
        super(MDESiamDataset, self).__init__()
        self.transforms = transforms
        self.pair_per_seqs = pair_per_seqs

        # 读取数据集所包含的视频序列，元数据，噪声标签，目标在图像中的宽高比例。

        # 判断是否存在元数据，如果存在直接读取，否则构建元数据的缓存，方便之后快速调用。
        if os.path.isfile(r'..\cache\meta_data.pickle'):
            with open(r'..\cache\meta_data.pickle', 'rb') as f:
                cache_meta_data = pickle.load(f)
            seqs = cache_meta_data['seqs']
            self.seqs = seqs
            meta_data = cache_meta_data['meta_data']
            noisy_label = cache_meta_data['noisy_label']
            target_wh = cache_meta_data['target_wh']
        else:
            seqs, meta_data, noisy_label, target_wh = self._load_meta_data(dataset_paths)
            self.seqs = seqs

        # 利用目标所属的类别，将训练集分开。
        with open(r'../cluster_result/result_index.txt') as f:
            cluster_indexs = f.readlines()
        cluster_indexs = [int(x.replace('\n', '')) for x in cluster_indexs]
        cluster_indexs = np.fromiter(cluster_indexs, np.int32)

        self.cluster_num = cluster_indexs.max() # 获得簇的数量
        self.cluster_seqs = []

        # 为每个簇预分配元数据存储列表
        [self.cluster_seqs.append([]) for x in range(self.cluster_num)]
        self.cluster_meta_data = copy.deepcopy(self.cluster_seqs)
        self.cluster_noisy_label = copy.deepcopy(self.cluster_seqs)
        self.cluster_target_wh = copy.deepcopy(self.cluster_seqs)

        # 按照簇的方式存储元数据。
        for i, cluster_index in enumerate(cluster_indexs):
            self.cluster_seqs[cluster_index-1].append(self.seqs[i])
            self.cluster_meta_data[cluster_index-1].append(meta_data[i])
            self.cluster_noisy_label[cluster_index-1].append(noisy_label[i])
            self.cluster_target_wh[cluster_index-1].append(target_wh[i])

        self.indices = np.random.permutation(min([len(x) for x in self.cluster_seqs]))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]  # 获得视频索引。

        # 获得图像目录
        cluster_image_files = [[], []]
        # 获取视频目录下的图像。
        for i in range(self.cluster_num):
            image_path = self.cluster_seqs[i][index]
            # 根据不同的数据集单独获取图像
            if 'Cropped_ILSVRC2015' in image_path:
                img_files = glob(os.path.join(image_path, '*.JPEG'))

            if 'Cropped_GOT10K' in image_path:
                img_files = glob(os.path.join(image_path, '*.jpg'))

            if 'Cropped_LaSOT' in image_path:
                img_files = []
                for x in range(1, len(self.cluster_target_wh[i][index]) + 1):
                    img_name = '%8d' % x
                    img_name = img_name.replace(' ', '0') + '.jpg'
                    img_file = os.path.join(image_path, img_name)
                    img_files.append(img_file)
            if 'Cropped_TrackingNet' in image_path:
                img_files = glob(os.path.join(image_path, '*.jpg'))

            noisy_label = self.cluster_noisy_label[i][index]
            val_indices = np.logical_and.reduce(noisy_label)
            val_indices = np.where(val_indices)[0]
            rand_z, rand_x = self._sample_pair((val_indices))

            # 读取视频帧，并根据要求的变换对视频帧进行变换
            z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
            x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
            z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.transforms is not None:
                [z, x] = self.transforms(z, x)
            cluster_image_files[0].append(z)
            cluster_image_files[1].append(x)
        # 此处需要将张量在第一维度堆叠。
        cluster_image_files = [torch.stack(x, axis=0) for x in cluster_image_files]
        return cluster_image_files

    def __len__(self):
        return min([len(x) for x in self.cluster_seqs]) * self.pair_per_seqs

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):        # 进行最大100次的循环采样，如果采样得到的训练对之间的间隔低于100，则选用此训练对，如果超过采样次数仍然没有获得满足条件的样本对，则随机算则一个样本对。
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _load_meta_data(self, dataset_paths):
        # 构建数据集，并将其存储于缓存中，以方便下次直接调用，省掉生成元数据的时间。
        seqs = []
        with open(r'../cluster_result/result_root.txt') as f:
            seq_indexs = f.readlines()
        for seq_index in seq_indexs:
            if 'Cropped_ILSVRC2015' in seq_index:
                seqs.append(os.path.join(dataset_paths['ILSVRC15'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_GOT10K' in seq_index:
                seqs.append(os.path.join(dataset_paths['GOT10K'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_LaSOT' in seq_index:
                seqs.append(os.path.join(dataset_paths['LASOT'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_TrackingNet' in seq_index:
                seqs.append(
                    os.path.join(dataset_paths['TrackingNet'], seq_index.split('\\')[-1].replace('\n', '')))
        self.seqs = seqs

        # 加载视频序列的元数据
        meta_data = []
        meta_data_names = [os.path.join(x, 'meta_data.txt') for x in self.seqs]
        for meta_data_name in meta_data_names:
            with open(meta_data_name, 'rb') as f:
                meta_data.append(pickle.load(f))

        # 加载视频序列的标签
        noisy_label = []
        noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in self.seqs]
        for noisy_label_name in noisy_label_names:
            with open(noisy_label_name, 'rb') as f:
                noisy_label.append(pickle.load(f))

        # 加载目标在搜索图像中的长宽比例
        target_wh = []
        target_wh_names = [os.path.join(x, 'target_wh.txt') for x in self.seqs]
        for target_wh_name in target_wh_names:
            with open(target_wh_name, 'rb') as f:
                target_wh.append(pickle.load(f))
        # 将元数据存储到缓存中，方便下次加载。
        with open(r'../cache/meta_data.pickle', 'wb') as f:
            pickle.dump({'seqs': seqs, 'meta_data': meta_data, 'noisy_label': noisy_label, 'target_wh': target_wh},
                        f)
        return seqs, meta_data, noisy_label, target_wh


class MDESiamDataset_cluster(Dataset):
    """
         MDESiam train dataset for each cluster tracker. You shall put the cluster result into folder './cluster_result'
         first.
         cluster: return which cluster dataset. in range [0, 5] indecate which cluster current dataset is.
     """

    def __init__(self, dataset_paths, transforms=None,
                 cluster=1, pair_per_seqs=1):
        super(MDESiamDataset_cluster, self).__init__()
        self.transforms = transforms
        self.pair_per_seqs = pair_per_seqs
        self.cluster = cluster

        # 读取数据集所包含的视频序列，元数据，噪声标签，目标在图像中的宽高比例。

        # 判断是否存在元数据，如果存在直接读取，否则构建元数据的缓存，方便之后快速调用。
        if os.path.isfile(r'..\cache\meta_data.pickle'):
            with open(r'..\cache\meta_data.pickle', 'rb') as f:
                cache_meta_data = pickle.load(f)
            seqs = cache_meta_data['seqs']
            self.seqs = seqs
            meta_data = cache_meta_data['meta_data']
            noisy_label = cache_meta_data['noisy_label']
            target_wh = cache_meta_data['target_wh']
        else:
            seqs, meta_data, noisy_label, target_wh = self._load_meta_data(dataset_paths)
            self.seqs = seqs

        # 利用目标所属的类别，将训练集分开。
        with open(r'../cluster_result/result_index.txt') as f:
            cluster_indexs = f.readlines()
        cluster_indexs = [int(x.replace('\n', '')) for x in cluster_indexs]
        cluster_indexs = np.fromiter(cluster_indexs, np.int32)

        self.cluster_num = cluster_indexs.max()  # 获得簇的数量
        self.cluster_seqs = []

        # 为每个簇预分配元数据存储列表
        [self.cluster_seqs.append([]) for x in range(self.cluster_num)]
        self.cluster_meta_data = copy.deepcopy(self.cluster_seqs)
        self.cluster_noisy_label = copy.deepcopy(self.cluster_seqs)
        self.cluster_target_wh = copy.deepcopy(self.cluster_seqs)

        # 按照簇的方式存储元数据。
        for i, cluster_index in enumerate(cluster_indexs):
            self.cluster_seqs[cluster_index - 1].append(self.seqs[i])
            self.cluster_meta_data[cluster_index - 1].append(meta_data[i])
            self.cluster_noisy_label[cluster_index - 1].append(noisy_label[i])
            self.cluster_target_wh[cluster_index - 1].append(target_wh[i])

        self.indices = np.random.permutation(len(self.cluster_seqs[self.cluster]))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]  # 获得视频索引。

        # 获得图像目录
        cluster_image_files = [[], []]
        # 获取视频目录下的图像。
        image_path = self.cluster_seqs[self.cluster][index]
        # 根据不同的数据集单独获取图像
        if 'Cropped_ILSVRC2015' in image_path:
            img_files = glob(os.path.join(image_path, '*.JPEG'))

        if 'Cropped_GOT10K' in image_path:
            img_files = glob(os.path.join(image_path, '*.jpg'))

        if 'Cropped_LaSOT' in image_path:
            img_files = []
            for x in range(1, len(self.cluster_target_wh[self.cluster][index]) + 1):
                img_name = '%8d' % x
                img_name = img_name.replace(' ', '0') + '.jpg'
                img_file = os.path.join(image_path, img_name)
                img_files.append(img_file)
        if 'Cropped_TrackingNet' in image_path:
            img_files = glob(os.path.join(image_path, '*.jpg'))

        noisy_label = self.cluster_noisy_label[self.cluster][index]
        val_indices = np.logical_and.reduce(noisy_label)
        val_indices = np.where(val_indices)[0]
        rand_z, rand_x = self._sample_pair((val_indices))

        # 读取视频帧，并根据要求的变换对视频帧进行变换
        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        item = (z, x)
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.indices) * self.pair_per_seqs

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):  # 进行最大100次的循环采样，如果采样得到的训练对之间的间隔低于100，则选用此训练对，如果超过采样次数仍然没有获得满足条件的样本对，则随机算则一个样本对。
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _load_meta_data(self, dataset_paths):
        # 构建数据集，并将其存储于缓存中，以方便下次直接调用，省掉生成元数据的时间。
        seqs = []
        with open(r'../cluster_result/result_root.txt') as f:
            seq_indexs = f.readlines()
        for seq_index in seq_indexs:
            if 'Cropped_ILSVRC2015' in seq_index:
                seqs.append(os.path.join(dataset_paths['ILSVRC15'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_GOT10K' in seq_index:
                seqs.append(os.path.join(dataset_paths['GOT10K'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_LaSOT' in seq_index:
                seqs.append(os.path.join(dataset_paths['LASOT'], seq_index.split('\\')[-1].replace('\n', '')))
            if 'Cropped_TrackingNet' in seq_index:
                seqs.append(
                    os.path.join(dataset_paths['TrackingNet'], seq_index.split('\\')[-1].replace('\n', '')))
        self.seqs = seqs

        # 加载视频序列的元数据
        meta_data = []
        meta_data_names = [os.path.join(x, 'meta_data.txt') for x in self.seqs]
        for meta_data_name in meta_data_names:
            with open(meta_data_name, 'rb') as f:
                meta_data.append(pickle.load(f))

        # 加载视频序列的标签
        noisy_label = []
        noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in self.seqs]
        for noisy_label_name in noisy_label_names:
            with open(noisy_label_name, 'rb') as f:
                noisy_label.append(pickle.load(f))

        # 加载目标在搜索图像中的长宽比例
        target_wh = []
        target_wh_names = [os.path.join(x, 'target_wh.txt') for x in self.seqs]
        for target_wh_name in target_wh_names:
            with open(target_wh_name, 'rb') as f:
                target_wh.append(pickle.load(f))
        # 将元数据存储到缓存中，方便下次加载。
        with open(r'../cache/meta_data.pickle', 'wb') as f:
            pickle.dump({'seqs': seqs, 'meta_data': meta_data, 'noisy_label': noisy_label, 'target_wh': target_wh},
                        f)
        return seqs, meta_data, noisy_label, target_wh






