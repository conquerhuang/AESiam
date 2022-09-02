# # 生成ILSVRC2015的list文件。
# import os
#
# Cropped_ILSVRC2015_dir = r'G:\dataset\train_dataset\Cropped_ILSVRC2015'
#
# files = os.listdir(Cropped_ILSVRC2015_dir)
# files = [os.path.join(Cropped_ILSVRC2015_dir, x) for x in files]
# sequences = []
# for file in files:
#     if os.path.isdir(file):
#         sequences.append(file)
#
# sequence_names = []
# for sequence in sequences:
#     sequence_names.append(os.path.basename(sequence) + '\n')
#
# with open(os.path.join(Cropped_ILSVRC2015_dir, 'list.txt'), 'w') as f:
#     f.writelines(sequence_names)

# 获取got10k和ILSVRC2015的noisylabel

# import scipy.io as scio
# from siamfc.datasets import Got10kCropped, ILSVRC2015Cropped
# import os
#
# root_dir_got10k = os.path.expanduser(r'G:\dataset\train_dataset\Cropped_GOT10K')
# root_dir_ilsvrc2015 = os.path.expanduser(r'G:\dataset\train_dataset\Cropped_ILSVRC2015')
#
# dataset_got10k = Got10kCropped(root_dir_got10k)
# dataset_ilsvrc2015 = ILSVRC2015Cropped(root_dir_ilsvrc2015)
#
# scio.savemat('noisy_label_got10k.mat', {'noisy_label_got10k': dataset_got10k.noisy_label})
# scio.savemat('noisy_label_ilsvrc15.mat', {'noisy_label_ilsvrc15': dataset_ilsvrc2015.noisy_label})


# # 生成LaSOT的list
# import os
# LaSOT_dir = r'G:\dataset\train_dataset\Cropped_LaSOT'
# sequence_dirs = os.listdir(LaSOT_dir)
# sequences = []
# for sequence_dir in sequence_dirs:
#     if os.path.isdir(os.path.join(LaSOT_dir, sequence_dir)):
#         sequences.append(sequence_dir)
# sequences = [x+'\n' for x in sequences]
# with open(os.path.join(LaSOT_dir, 'list.txt'), 'w') as f:
#     f.writelines(sequences)


# 比较训练集中的视频帧数。

from siamfc.datasets import Got10kCropped, ILSVRC2015Cropped, LaSOTCropped

ilsvrc_dir = r'G:\dataset\train_dataset\Cropped_ILSVRC2015'
got1ok_dir = r'G:\dataset\train_dataset\Cropped_GOT10K'
lasot_dir = r'G:\dataset\train_dataset\Cropped_LaSOT'

ilsvrc_dataset = ILSVRC2015Cropped(ilsvrc_dir, pair_per_seq=1)
got1ok_dataset = Got10kCropped(got1ok_dir, pair_per_seq=1)
lasot_dataset = LaSOTCropped(lasot_dir, pair_per_seq=1)

ilsvrc_frames = 0
go10k_frames = 0
lasot_frames = 0
for i in range(len(ilsvrc_dataset)):
    sequence_length = len(ilsvrc_dataset.target_wh[i])
    ilsvrc_frames += sequence_length
print('dataset: ilsvrc15  sequences:{}  frames:{}'.format(len(ilsvrc_dataset), ilsvrc_frames))

for i in range(len(got1ok_dataset)):
    sequence_length = len(got1ok_dataset.target_wh[i])
    go10k_frames += sequence_length
print('dataset: got10k  sequences:{}  frames:{}'.format(len(got1ok_dataset), go10k_frames))

for i in range(len(lasot_dataset)):
    sequence_length = len(lasot_dataset.target_wh[i])
    lasot_frames += sequence_length

print('dataset: lasot  sequences:{}  frames:{}'.format(len(lasot_dataset), lasot_frames))









