%{
Code by Kellen Vu.

This program uses an LSTM network to identify saccades vs. non-saccades.
%}

%% Options

isTraining = false; % Set to false if you already trained the network

%% Load data

[XMerged, YMerged] = mergeData(1000);

XTrain = XMerged(1:4000);
YTrain = YMerged(1:4000);
XTest = XMerged(4001:end);
YTest = YMerged(4001:end);

% Plot one example
%{
X = XTrain{1};
classes = categories(YTrain{1});

figure
for i = 1:numel(classes)
    label = classes(i);
    idx = find(YTrain{1} == label);
    plotVal = NaN(size(X));
    plotVal(idx) = X(idx);
    hold on
    plot(plotVal)
end

xlabel('Time (ms)')
ylabel('Position (deg)')
title('Example 1')
legend(classes)
%}

%% Train

miniBatchSize = 64;
numFeatures = 1;
numHiddenUnits = 200;
numClasses = 2;
if isTraining
    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    
    options = trainingOptions('adam', ...
        'Verbose', 0, ...
        'Plots', 'training-progress', ...
        'MiniBatchSize', miniBatchSize);
    
    [net, info] = trainNetwork(XTrain, YTrain, layers, options);
    save('net.mat', 'net', 'info')
else
    load('net.mat')
end

%% Test Plot

X = cat(2, XTrain{1500:1600});
Y = cat(2, YTrain{1500:1600});
classes = categories(Y);

YPred = classify(net, X);

figure
tiledlayout(2, 1)
ax1 = nexttile;
plot(YPred, '.-')
hold on
plot(Y)
hold off

xlabel('Time step')
title('Predicted Classes')
legend('Predicted', 'Ground truth')

ax2 = nexttile;
for i = 1:numel(classes)
    label = classes{i};
    idx = find(Y == label);
    plotVal = NaN(size(X));
    plotVal(idx) = X(idx);
    hold on
    if strcmp(label, 'Non-saccade')
        plot(plotVal, 'Color', [0.7, 0.7, 0.7])
    else
        plot(plotVal, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1)
    end
end

xlabel('Time (ms)')
ylabel('Position (deg)')
title('Ground Truth')
legend(classes)
linkaxes([ax1, ax2], 'x')

%% Evaluation

truePos = 0;
actualPos = 0;
N = 400;
for i = 1:9
    X = cat(2, XTrain{i * 400 + 1:(i + 1) * 400});
    YPred = classify(net, X, 'MiniBatchSize', miniBatchSize);
    Y = cat(2, YTrain{i * 400 + 1:(i + 1) * 400});
    truePos = truePos + sum((YPred == Y) & (Y == 'Saccade'));
    actualPos = actualPos + sum(Y == 'Saccade');
end

fprintf('Train Set: ClassifiedAsSaccadeByNN / ClassifiedAsSaccadeByThreshold = %d\n', truePos / actualPos)

beep

%% Functions

function [XMerged, YMerged] = mergeData(exampleSize)
    % Merge multiple data.mat files into one X cell array and one Y cell array.
    % :param exampleSize: The maximum length of each example
    % :return XMerged: The merged input data (cell array)
    % :return YMerged: The merged label data (cell array)
    XMerged = {};
    YMerged = {};
    
    % Load .mat files
    dataFiles = dir('data/*_data.mat');
    for i = 1:length(dataFiles)
        file = fullfile(dataFiles(i).folder, dataFiles(i).name);
        load(file, 'X', 'Y')
        
        % Split each file into multiple examples (to make it easier on the GPU)
        N = floor(numel(X) / exampleSize);
        X = reshape(X(1:N * exampleSize), exampleSize, [])';
        X = num2cell(X, 2);
        Y = reshape(Y(1:N * exampleSize), exampleSize, [])';
        Y = num2cell(Y, 2);

        XMerged = [XMerged; X];
        YMerged = [YMerged; Y];
    end
end