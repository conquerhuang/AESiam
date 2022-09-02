clear
clc

% ��Ƶ����Ŀ¼
feat_dir = 'F:\MDESiam\siamvgg\dataset_feature_oc_24';

% ��ȡĿ¼�����е���Ƶ����
feat_files = dir(feat_dir);
feat_files = feat_files(3:end);
feat_names = {feat_files(:).name};

video_features = zeros(length(feat_names), 5*5*24);
video_features_root = strings(length(feat_names), 1);
for i = 1:1:length(feat_names)
    load([feat_dir, '\\', feat_names{i}])
    
    % ������ȥ���Ĳ����򻯡�
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



