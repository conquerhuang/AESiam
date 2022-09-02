% get data from file
load('result_compare.mat')

nbins = 50;

% acquire cluster number and cluster sum error from data
num_trad = result_compare.num{1};
num_bal = result_compare.num{2};

err_trad = result_compare.err{1};
err_bal = result_compare.err{2};

% drow plot about cluster number distribution
figure
yyaxis left  % drow traditioal num distribution plot
[N, edges]=histcounts(num_trad, nbins);
bar((edges(1:end-1)+edges(2:end))/2, N, 'edgecolor', 'none')

yyaxis right % drow balanced num distribution plot
[N, edges]=histcounts(num_bal, nbins);
bar((edges(1:end-1)+edges(2:end))/2, N, 'edgecolor', 'none')
xlabel('number of samples per cluster','fontsize', 24)
title('cluster sample numble distribution', 'fontsize', 24)
legend({'traditional', 'balanced'}, 'fontsize', 24)
set(gca, 'fontsize', 24)



% drow plot about error distribution
figure
yyaxis left % draw traditional error distribution plot
[N, edges]=histcounts(err_trad, nbins);
bar((edges(1:end-1)+edges(2:end))/2, N, 'edgecolor', 'none')

yyaxis right % draw balanced error distribution plot
[N, edges]=histcounts(err_bal, nbins);
bar((edges(1:end-1)+edges(2:end))/2, N, 'edgecolor', 'none')
xlabel('sum error', 'fontsize', 24)
title('sum error distribution','fontsize', 24)
legend({'traditional','balanced'}, 'fontsize', 24)
set(gca, 'fontsize', 24)





