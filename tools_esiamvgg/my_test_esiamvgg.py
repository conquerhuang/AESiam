import os
from got10k.experiments import *
from siamfc import TrackerESiamVgg

from multiprocessing import Pool


def evaluate(model_path, tracker_name, experiment):
    tracker = TrackerESiamVgg(net_path=model_path, name=tracker_name)
    # for experiment in experiments:
    experiment.run(tracker)
    experiment.report([tracker_name])


if __name__ == '__main__':
    experiments = [
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2015),
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2013),
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='tb50'),
        ExperimentDTB70(root_dir=r'E:\dataset\tracker_evaluate_dataset\DTB70'),
        ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128'),
        # ExperimentGOT10k(root_dir=r'E:\dataset\train_dataset\GOT-10K\full_data'),
        ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123'),
        ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV20L')
        # ExperimentLaSOT(root_dir='')
    ]

    # 单线程
    # for i in range(21, 51):
    #     model_path = r'../pretrained/SiamFC_new/siam_vggnet_e'+ str(i) + '.pth'
    #     tracker_name = 'SiamVGG_' + str(i)
    #     tracker = TrackerSiamvgg(model_path=model_path, tracker_name=tracker_name) #初始化一个追踪器
    #     for experiment in experiments:
    #         experiment.run(tracker)
    #         experiment.report(tracker_name)

    # # 多线程, 多模型
    # tracker_infos = []
    # for i in range(10, 50):
    #     model_path = r'../pretrained/ESiamvgg_'+ str(i) + '.pth'
    #     tracker_name = 'Siamvgg_' + str(i)
    #     for experiment in experiments:
    #         tracker_infos.append([model_path, tracker_name, experiment])
    # p = Pool(processes=3)
    # for tracker_info in tracker_infos:
    #     p.apply_async(evaluate, args=tuple(tracker_info))
    # p.close()
    # p.join()

    # 多线程, 单个模型
    tracker_infos = []

    # model_path = r'./transformed_esiamvgg.pth'
    model_path = r'../pretrained/EsiamVGG_19.PTH'
    tracker_name = 'ESiamvgg'
    for experiment in experiments:
        tracker_infos.append([model_path, tracker_name, experiment])
    p = Pool(processes=3)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()


    # model_path = r'../pretrained/SiamFC/siamfc_vgg.pth'
    # tracker_name = 'SiamFC_vgg'
    # for experiment in experiments:
    #     evaluate(model_path, tracker_name, experiment)
