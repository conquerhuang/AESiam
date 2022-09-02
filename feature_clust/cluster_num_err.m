function [loss] = cluster_num_err(class_num)
    loss = (class_num - mean(class_num))/mean(class_num);
    loss = map_function(loss);
end

function [x] = map_function(x)
    index = x<0;
    x(index) = x(index)*5;
end