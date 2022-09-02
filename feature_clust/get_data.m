clear
clc

% 视频特征目录
feat_dir = 'F:\MDESiam\siamvgg\dataset_feature_oc_24';

% 获取目录下所有的视频特征
feat_files = dir(feat_dir);
feat_files = feat_files(3:end);
feat_names = {feat_files(:).name};

video_features = zeros(length(feat_names), 5*5*24);
video_features_root = strings(length(feat_names), 1);
for i = 1:1:length(feat_names)
    load([feat_dir, '\\', feat_names{i}])
    
    % 将数据去中心并正则化。
    feature = feature(:);
    feature = feature-mean(feature);
    feature = feature./(sqrt(feature' * feature));
    
    video_features(i,:) = feature(:);
    video_features_root(i, 1) = root_dir;
    if mod(i, 100)==0
        fprintf('%.1f\n',i/length(feat_names))
    end
end

[coeff, score, lattent] = pca(video_features);
features_pca = score(:, 1:3);
clear coeff feat_dir feat_files feat_names feature i lattent root_dir score



