figure
plot3(features_pca(:, 1), features_pca(:, 2), features_pca(:, 3), 'b.','markersize', 8)
grid on
set(gca, 'XMinorGrid','on');
set(gca, 'YMinorGrid','on');
set(gca, 'ZMinorGrid','on');
title('training samples in embedded space', 'FontSize', 24)
view(45.029999881888919,37.568826669688846)
set(gca, 'fontsize', 24)
set(gcf, 'position', [200, 100, 800, 700])



figure
hold off
for i = 1:k
    class_index = video_features_class_new == i; %获取当前类别索引
    plot3(features_pca(class_index,1),features_pca(class_index,2),...
        features_pca(class_index,3),[colors(i) '.'], 'markersize', 16)
    if i == 1
        hold on
    end
end
grid on
set(gca, 'XMinorGrid','on');
set(gca, 'YMinorGrid','on');
set(gca, 'ZMinorGrid','on');
view(45.029999881888919,37.568826669688846)
set(gca, 'fontsize', 24)
set(gcf, 'position', [200, 100, 800, 700])
title('clustering result', 'fontsize', 24)












