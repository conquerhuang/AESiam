import torch
from siamfc import TrackerESiamVgg
from siamfc import TrackerSiamvgg_cluster

if __name__ == '__main__':
    cluster1_path = r'../pretrained/cluster0_12.pth'
    cluster2_path = r'../pretrained/cluster1_14.pth'
    cluster3_path = r'../pretrained/cluster2_16.pth'
    cluster4_path = r'../pretrained/cluster3_37.pth'
    cluster5_path = r'../pretrained/cluster4_13.pth' 
    cluster6_path = r'../pretrained/cluster5_17.pth'

    tracker_cluster = TrackerSiamvgg_cluster()
    tracker = TrackerESiamVgg()

    # load dict of clusters.
    tracker_cluster.net.load_state_dict(torch.load(cluster1_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster1.load_state_dict(dict_cluster)

    tracker_cluster.net.load_state_dict(torch.load(cluster2_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster2.load_state_dict(dict_cluster)

    tracker_cluster.net.load_state_dict(torch.load(cluster3_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster3.load_state_dict(dict_cluster)

    tracker_cluster.net.load_state_dict(torch.load(cluster4_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster4.load_state_dict(dict_cluster)

    tracker_cluster.net.load_state_dict(torch.load(cluster5_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster5.load_state_dict(dict_cluster)

    tracker_cluster.net.load_state_dict(torch.load(cluster6_path))
    dict_cluster = tracker_cluster.net.backbone.cluster_feature.state_dict()
    tracker.net.backbone.cluster6.load_state_dict(dict_cluster)

    # load dict of share features.
    dict_stem = tracker_cluster.net.backbone.share_features.state_dict()
    tracker.net.backbone.share_features.load_state_dict(dict_stem)

    # load dict of head.
    dict_head = tracker_cluster.net.head.state_dict()
    tracker.net.head.load_state_dict(dict_head)

    torch.save(tracker.net.state_dict(), 'transformed_esiamvgg.pth')






