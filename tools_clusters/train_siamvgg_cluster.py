
from siamfc import TrackerSiamvgg_cluster
from siamfc import MDESiamDataset_cluster

if __name__ == '__main__':
    dataset_paths = {
        'ILSVRC15': r'F:\Cropped_ILSVRC2015',
        'GOT10K': r'F:\Cropped_GOT10K',
        'LASOT': r'F:\Cropped_LaSOT',
        'TrackingNet': r'F:\Cropped_TrackingNet'
    }

    net_path = r'../pretrained/SiamFC/transformed_model_siamvgg_oc24.pth'
    num_epoch = 50
    pair_per_seqs = 1

    for cluster in range(0, 6):
        train_dataset = MDESiamDataset_cluster(
            dataset_paths=dataset_paths, 
            pair_per_seqs=num_epoch * pair_per_seqs,
            cluster=cluster
        )

        tracker_name = 'cluster' + str(cluster)

        tracker = TrackerSiamvgg_cluster(net_path=net_path, name=tracker_name)

        tracker.train_over(train_dataset)


