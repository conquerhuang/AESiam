# Attention Ensemble Siamese Networks for object tracking
Ensemble Siamese networks improve the tracking performance by integrating diverse base trackers into one robust tracker. Despite the success, this framework suffers from a cumbersome model and slow tracking speed, and the way to aggregate base trackers into one powerful tracker remains challenging. We propose a novel attention ensemble Siamese network AESiam to solve these problems. The proposed AESiam comprises two simple modules: one feature attention module to compress the base trackers and one channel attention module for base tracker integration. Specifically, we integrate the feature attention module into the convolutional layer to learn the hidden internal relationship. Then, we decouple the convolutional layer by the learned relationship to achieve compact base trackers. To alleviate the hard-to-aggregate problem, we expand the cluster weight fusion strategy of the Ensemble Siamese networks with a channel attention module. The channel attention module selectively emphasizes informative feature channels according to the tracking scenario and integrates base trackers into a robust tracker. Extensive experiments on 6 benchmarks demonstrate that our AESiam achieves a significant increase in tracking performance and speed compared to the baseline tracker and performs as a counterpart of other recent trackers.

# Environment configuration
    The environment required for this project is given in the require.txt file.

# Training AESiam
## prepare training dataset
1. Download the got10k dataset from http://got-10k.aitestunion.com/index
2. run \prepare_datasets\crop_train_dataset_got10k.py to crop the training dataset.
3. run \prepare_datasets\gen_meta_data_pickle.py to generate the corresponding mate data of the got10k dataset, and move it into the folder where the cropped dataset is.

## Train the SiamVgg (we reformulate the training code of the SiamVgg tracker)
We reconstruct the training code of SimaVgg (https://github.com/leeyeehoo/SiamVGG) in a very simple and clean form.
1. Run  .\tools\train_siamvgg_oc.py to train the SiamVgg.
2. Run .\tools\my_test_evgg_oc.py to test all the trained models and select the best one for the subsequent work
3. Run .\tools\model_transform.py to decouple the convolutional layers of the cluster backbone.

## Mapping the training dataset into the Embedded space of the trained backbone of the previous step.
1. Open the .\acquire_video_feature\video_feature.py file and modify the  dataset_paths, video_feature_dir and, net_path.
2. Run .\acquire_video_feature\feature_embedding.py to map the training dataset into the embedded space.

## Clustering the training dataset in the embedded space.
1. Switch to the Matlab IDE and set the feature_clust folder as the work folder.
2. Run .\feature_clust\get_data.m to load the mapped dataset into the workspace.
3. Run .\feature_clust\demo.m to cluster the training samples. If the clustering process does not converge, finetune the control_i and the control_p parameters.
4. Run .\feature_clust\create_result.m file to generate the result files 'result_index.txt', 'result_root.txt'

## Training base trackers.
1. Moving the cluster result files  'result_index.txt', 'result_root.txt' to the ./cluster_result folder.
2. Run .\tools_clusters\train_siamvgg_cluster.py to train base trackers.
3. Run .\tools_clusters\my_test_siamvgg_cluster.py to test the base trackers in each cluster and select the best one in each cluster as the base trackers in subsequent steps.


## Integrate the base trackers into one ensemble tracker (without fusion weight)
1. Modify the path of the base trackers in .\tools_esiamvgg\transform_model.py.
2. Run .\tools_esiamvgg\transform_model.py to integrate the base trackers into one ensemble tracker.

## Training the channel attention module.
1. Run .\tools_channel_attention\model_transform.py file to integrate the channel attention module into the ensemble tracker.
2. Run .\tools_channel_attention\train_esiamvgg_channel_attention.py to train the channel attention module.
3. Run .\ tools_channel_attention\my_test_esiamatten.py to test the trained channel attention module to select the best model.

## Hyperparameter setting
1. The hyperparameters of AESiam saved at the parse_args member function the class TrackerESiamVggAtten class in ./siamfc/esiamvgg_atten.py finetune them to get the best parameters.
2. The parameter for each benchmark is saved in the .\hyper_parameter\config_datasets folder.

# Test the AESiam
1. Download the pretrained module at https://pan.baidu.com/s/1bwBod5R4wCqK1PC4I13jgQ?pwd=ag35 keyword：ag35.
2. Open .\tools_channel_attention\demo.py file and set the model path
3. Run .\ tools_channel_attention\demo.py

