import torch
from siamfc import TrackerESiamVgg, TrackerESiamVggAtten

if __name__ == '__main__':
    esiam_path = r'../tools_esiamvgg/transformed_esiamvgg.pth'

    dict_esiamvgg = torch.load(esiam_path)

    tracker_atten = TrackerESiamVggAtten()
    atten_dict = tracker_atten.net.state_dict()

    tracker_dict = {}
    for k, v in dict_esiamvgg.items():
        if 'share_features' in k:
            tracker_dict[k] = v
        elif 'head' in k:
            tracker_dict[k] = v
        else:
            pass

    cluster1_w = dict_esiamvgg['backbone.cluster1.0.weight']
    cluster1_b = dict_esiamvgg['backbone.cluster1.0.bias']

    cluster2_w = dict_esiamvgg['backbone.cluster2.0.weight']
    cluster2_b = dict_esiamvgg['backbone.cluster2.0.bias']

    cluster3_w = dict_esiamvgg['backbone.cluster3.0.weight']
    cluster3_b = dict_esiamvgg['backbone.cluster3.0.bias']

    cluster4_w = dict_esiamvgg['backbone.cluster4.0.weight']
    cluster4_b = dict_esiamvgg['backbone.cluster4.0.bias']

    cluster5_w = dict_esiamvgg['backbone.cluster5.0.weight']
    cluster5_b = dict_esiamvgg['backbone.cluster5.0.bias']

    cluster6_w = dict_esiamvgg['backbone.cluster6.0.weight']
    cluster6_b = dict_esiamvgg['backbone.cluster6.0.bias']

    clusters_w = torch.cat(
        (cluster1_w, cluster2_w, cluster3_w, cluster4_w, cluster5_w, cluster6_w),
        dim=0
    )
    clusters_b = torch.cat(
        (cluster1_b, cluster2_b, cluster3_b, cluster4_b, cluster5_b, cluster6_b),
        dim=0
    )
    tracker_dict['backbone.clusters.0.weight'] = clusters_w
    tracker_dict['backbone.clusters.0.bias'] = clusters_b

    atten_dict.update(tracker_dict)

    tracker_atten.net.load_state_dict(atten_dict)
    torch.save(tracker_atten.net.state_dict(), 'transformed_atten_dict.pth')
