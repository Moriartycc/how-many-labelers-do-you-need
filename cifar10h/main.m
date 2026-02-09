load('outputs.mat');
load('pred.mat');
load('true.mat');

rng('default')

d = 40;
label = 7;
m = 10; % ten-fold cross validation
p_index = 0.8:0.005:0.95;
p_acc_nn = zeros(1, length(p_index));
p_calibacc_nn = p_acc_nn;
p_acc_logistic = p_acc_nn;
p_calibacc_logistic = p_acc_nn;
p_acc_maj = p_acc_nn;
p_calibacc_maj = p_acc_nn;
p_n = p_acc_nn;
j = 0;
%  I = true_class ~= label - 1;
%  for k = 1 : 10000
%     if sum(I(1:k)) == sum(true_class == label - 1)
%        break;
%     end
%  end
%  I = (true_class == label - 1) | (I & [ones(1,k), zeros(1, 10000-k)]);
%  I = I(randperm(length(I)));
%  outputs = outputs(I, :);
%  true_class = true_class(I);
%  true_count = true_count(I, :);
%  true_calib = true_calib(I, :);
%  pred_class = pred_class(I);
%  pred_calib = pred_calib(I, :);
outputs = double(outputs);
[~, S, V] = svds(outputs, d);
outputs = outputs * V;
outputs = normalize(outputs);
outputs = [ones(length(outputs), 1), outputs];
cvx_solver Mosek
cvx_precision high
for p = p_index
    j = j + 1; 
    p
    I = max(sum(true_calib(:, label), 2), 1 - sum(true_calib(:, label), 2)) <= p;
    % I = max(true_calib, [], 2) <= p;
    n = sum(I);
    p_n(j) = n;
    true_count = double(true_count);
    t_count = true_count(I, :);
    out = outputs(I, :);
    t_class = true_class(I)';
    t_calib = true_calib(I, :);
    p_class = pred_class(I)';
    p_calib = pred_calib(I, :);
    acc_logistic = 0;
    calibacc_logistic = 0;
    acc_maj = 0;
    calibacc_maj = 0;

    %% Neural network accuracy and calibration error
    acc_nn = sum(xor(sum(p_calib(:, label), 2) > 0.5, sum(t_class == label - 1, 2))) / n;
    calibacc_nn = 2 * norm(sum(p_calib(:, label), 2) - sum(t_calib(:, label), 2), 1) / n;

    for k = 1:m
        train_size = n - (round(k*n/m) - round((k-1)*n/m));
        test_size = n - train_size;
        I_test = round((k-1)*n/m)+1:round(k*n/m);
        I_train = [1:round((k-1)*n/m), round(k*n/m)+1:n];


        %% Binary classification
        X = zeros(2 * train_size, d+1);
        weight = zeros(2 * train_size, 1);
        weight_maj = weight;
        X(1:2:2*train_size-1, :) = -out(I_train, : );
        X(2:2:2*train_size, :) = out(I_train, : );
        weight(1:2:2*train_size-1, :) = sum(t_count(I_train, label), 2);
        weight(2:2:2*train_size, :) = sum(t_count(I_train, :), 2) - sum(t_count(I_train, label), 2);
        % weight_maj(1:2:2*train_size, :) = rand(1) < (sum(t_count(I_train, label), 2) ./ sum(t_count(I_train, :), 2));
        weight_maj(1:2:2*train_size-1, :) = (sign(weight(1:2:2*train_size-1, :) - weight(2:2:2*train_size, :)) + 1) / 2;
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

        %% Logistic regression accuracy and calibration error
        logistic_calib = 1./(1 + exp(-out(I_test,:) * theta));
        acc_logistic = acc_logistic + sum(xor(logistic_calib > 0.5, sum(t_class(I_test) == label - 1, 2))) / test_size / m;
        calibacc_logistic = calibacc_logistic + (2 *norm(logistic_calib - sum(t_calib(I_test, label), 2), 1) / test_size) / m;
        maj_calib = 1./(1 + exp(-out(I_test,:) * theta_maj));
        acc_maj = acc_maj + sum(xor(maj_calib > 0.5, sum(t_class(I_test) == label - 1, 2))) / test_size / m;
        calibacc_maj = calibacc_maj + (2 *norm(maj_calib - sum(t_calib(I_test, label), 2), 1) / test_size) / m;
    end
    p_acc_nn(j) = acc_nn;
    p_calibacc_nn(j) = calibacc_nn;
    p_acc_logistic(j) = acc_logistic;
    p_calibacc_logistic(j) = calibacc_logistic;
    p_acc_maj(j) = acc_maj;
    p_calibacc_maj(j) = calibacc_maj;
end
figure
plot(p_index, p_acc_nn, '-.', p_index, p_acc_logistic, '-o', p_index, p_acc_maj, '-o', 'Markersize', 3);
xlabel('$$p$$', 'interpreter', 'latex');
ylabel('Classification error');
legend('Pretrained network', 'Logistic regression on individual vote', 'Logistic regression on majority vote');
h = figure(1);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h, '-dpdf', strcat('./resnet-class.pdf'));

figure
plot(p_index, p_calibacc_nn, '-.', p_index, p_calibacc_logistic, '-o', p_index, p_calibacc_maj, '-o', 'Markersize', 3);
xlabel('$$p$$', 'interpreter', 'latex');
ylabel('Calibration error');
legend('Pretrained network', 'Logistic regression on individual vote', 'Logistic regression on majority vote');
h = figure(2);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h, '-dpdf', strcat('./resnet-calib.pdf'));