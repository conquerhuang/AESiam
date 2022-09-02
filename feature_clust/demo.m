% 获取用于聚类的数据，返回两个值
% 返回值：  
% video_features： 用于聚类的特征，每一行为一个特征
% video_features_root 每个特征对应的存储路径

% load('matlab.mat')
% data_cluster_balanced_min2
clc
clear
close all

control_p = 0.005;
control_i = 0.003;
k = 6;

load('matlab.mat')
video_features = gpuArray(video_features);
data_cluster_balanced_min2
figure
plot(accu_err)

result.index = video_features_class_new;
result.root = video_features_root;
% save('cluster_result.mat', 'result')



% for k = 2:1:15
%     control_p = 0.002;
%     control_i = 0.001;
% 
%     % load('matlab.mat')
%     % load('video_features_pca.mat')
%     % % 50:89.9%; 100:97.0%; 150:99.0%; 200:99.66%; 250:99.87%; 300:99.95% 
%     % video_features = score(:, 1:200); 
%     load('matlab.mat')
%     video_features = gpuArray(video_features);
%     data_cluster_balanced_min2
% 
%     figure
%     plot(accu_err)
%     title(['k=' num2str(k) '  accu\_err:' num2str(accu_err(end))])
%     fprintf('k:%3d,  accu_err:%.4f\n', k, accu_err(end))
%     pause(1)
%     
%     clear
% end


