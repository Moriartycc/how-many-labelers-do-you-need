load('original.tsv');
load('resnet-features.txt');
load('vgg-features.txt');
[image_ID, image_first, image_all] = unique(original(:, 2));
[lbler_ID, lbler_first, lbler_all] = unique(original(:, 1));
n = length(image_ID);
M_index = 1:1:35;
M_acc_logistic = zeros(1, length(M_index));
M_acc_logistic_CI = M_acc_logistic;
M_calibacc_logistic = M_acc_logistic;
M_calibacc_logistic_CI = M_acc_logistic;
M_acc_maj = M_acc_logistic;
M_acc_maj_CI = M_acc_logistic;
M_calibacc_maj = M_acc_logistic;
M_calibacc_maj_CI = M_acc_logistic;
trial = 100;
d = 25;
m = 10;
true_label = original(image_first, 4);
one_vote = zeros(n, 1);
zero_vote = zeros(n, 1);
for i = 1:length(original)
    if original(i, 3) == 1
        one_vote(image_all(i)) = one_vote(image_all(i)) + 1;
    end
    if original(i, 3) == 0
        zero_vote(image_all(i)) = zero_vote(image_all(i)) + 1;
    end
end
true_calib = one_vote ./ (one_vote + zero_vote);
features = resnet_features(:, 2:length(resnet_features));
[~, S, V] = svds(features, d);
features = features * V;
features = normalize(features);
features = [ones(n, 1), features];

logit = @(x) log(x./(1-x));
        
cvx_solver Mosek
cvx_precision high
j = 0;
for M = M_index
    j = j + 1;
    acc_logistic = zeros(1, trial);
    calibacc_logistic = acc_logistic;
    acc_maj = acc_logistic;
    calibacc_maj = acc_logistic;
    
    for l = 1:trial
        M
        l
        lb_ID = randperm(39);
        I = zeros(length(original), 1);
        one_vote = zeros(n, 1);
        zero_vote = zeros(n, 1);
        for i = 1:M
            I = I | (lbler_all == lb_ID(i));
        end
        I_index = 1:length(original);
        for i = I_index(I)
            if original(i, 3) == 1
                one_vote(image_all(i)) = one_vote(image_all(i)) + 1;
            end
            if original(i, 3) == 0
                zero_vote(image_all(i)) = zero_vote(image_all(i)) + 1;
            end
        end


        for k = 1:m
            train_size = n - (round(k*n/m) - round((k-1)*n/m));
            test_size = n - train_size;
            I_test = round((k-1)*n/m)+1:round(k*n/m);
            I_train = [1:round((k-1)*n/m), round(k*n/m)+1:n];

            X = zeros(2 * train_size, d+1);
            weight = zeros(2 * train_size, 1);
            weight_maj = weight;
            X(1:2:2*train_size-1, :) = -features(I_train, : );
            X(2:2:2*train_size, :) = features(I_train, : );
            weight(1:2:2*train_size-1, :) = one_vote(I_train);
            weight(2:2:2*train_size, :) = zero_vote(I_train);
            weight_maj(1:2:2*train_size-1, :) = rand(1) <= ((sign(one_vote(I_train) - zero_vote(I_train)) + 1)/2);
            weight_maj(2:2:2*train_size, :) = 1 - weight_maj(1:2:2*train_size-1, :);

            cvx_begin quiet
                variable theta(d+1)
                expression loss
                loss = weight' * log(1 + exp(X * theta));
                minimize loss
            cvx_end

            cvx_begin quiet
                variable theta_maj(d+1)
                expression loss_maj
                loss_maj = weight_maj' * log(1 + exp(X * theta_maj));
                minimize loss_maj
            cvx_end

            logistic_calib = 1./(1 + exp(-features(I_test,:) * theta));
            acc_logistic(l) = acc_logistic(l) + sum(xor(logistic_calib > 0.5, true_label(I_test) > 0.5)) / test_size / m;
            calibacc_logistic(l) = calibacc_logistic(l) + (norm(min(max(features(I_test,:) * theta, -10), 10) - logit(true_calib(I_test)), 1) / test_size) / m;
            maj_calib = 1./(1 + exp(-features(I_test,:) * theta_maj));
            acc_maj(l) = acc_maj(l) + sum(xor(maj_calib > 0.5, true_label(I_test) > 0.5)) / test_size / m;
            calibacc_maj(l) = calibacc_maj(l) + (norm(min(max(features(I_test,:) * theta_maj, -10), 10) - logit(true_calib(I_test)), 1) / test_size) / m;
        end
    end
    M_acc_logistic(j) = mean(acc_logistic);
    M_acc_logistic_CI(j) = 0.96 * std(acc_logistic) / sqrt(trial);
    M_calibacc_logistic(j) = mean(calibacc_logistic);
    M_calibacc_logistic_CI(j) = 0.96 * std(calibacc_logistic) / sqrt(trial);
    M_acc_maj(j) = mean(acc_maj);
    M_acc_maj_CI(j) = 0.96 * std(acc_maj) / sqrt(trial);
    M_calibacc_maj(j) = mean(calibacc_maj);
    M_calibacc_maj_CI(j) = 0.96 * std(calibacc_maj) / sqrt(trial);
end

figure
hold on
fill([M_index, fliplr(M_index)], [M_acc_logistic + M_acc_logistic_CI, fliplr(M_acc_logistic - M_acc_logistic_CI)], [0.7, 0.7, 1]);
fill([M_index, fliplr(M_index)], [M_acc_maj + M_acc_maj_CI, fliplr(M_acc_maj - M_acc_maj_CI)], [1, 0.7, 0.7]);
p1 = errorbar(M_index, M_acc_logistic, M_acc_logistic_CI, '-o', 'Markersize', 3, 'Color', 'blue');
p2 = errorbar(M_index, M_acc_maj, M_acc_maj_CI, '-o', 'Markersize', 3, 'Color', 'red');

xlabel('$$M$$', 'interpreter', 'latex', 'Fontsize', 12);
ylabel('Classification error', 'Fontsize', 12);
legend([p1, p2], {'Logistic regression on multiple votes', 'Logistic regression on majority vote'}, 'interpreter', 'latex', 'Fontsize', 12);
h = figure(1);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h, '-dpdf', strcat('./resnet-class-bluebirds.pdf'));
savefig('resnet-class-bluebirds.fig');

figure
hold on
ylim([0, 10]);
fill([M_index, fliplr(M_index)], [M_calibacc_logistic + M_calibacc_logistic_CI, fliplr(M_calibacc_logistic - M_calibacc_logistic_CI)], [0.7, 0.7, 1]);
fill([M_index, fliplr(M_index)], [M_calibacc_maj + M_calibacc_maj_CI, fliplr(M_calibacc_maj - M_calibacc_maj_CI)], [1, 0.7, 0.7]);
p1 = errorbar(M_index, M_calibacc_logistic, M_calibacc_logistic_CI, '-o', 'Markersize', 3, 'Color', 'blue');
p2 = errorbar(M_index, M_calibacc_maj, M_calibacc_maj_CI, '-o', 'Markersize', 3, 'Color', 'red');
xlabel('$$M$$', 'interpreter', 'latex', 'Fontsize', 12);
ylabel('Calibration error', 'Fontsize', 12);
legend([p1, p2], {'Logistic regression on multiple votes', 'Logistic regression on majority vote'}, 'interpreter', 'latex', 'Location', 'northwest', 'Fontsize', 12);
h = figure(2);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h, '-dpdf', strcat('./resnet-calib-bluebirds.pdf'));
savefig('resnet-calib-bluebirds.fig');
save('bluebirds.mat', 'M_index', 'M_acc_logistic', 'M_acc_logistic_CI', 'M_acc_maj', 'M_acc_maj_CI', 'M_calibacc_logistic', 'M_calibacc_logistic_CI', 'M_calibacc_maj', 'M_calibacc_maj_CI');