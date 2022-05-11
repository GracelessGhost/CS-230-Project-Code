%{
Code by Kellen Vu.

This program uses an LSTM network to identify saccades vs. non-saccades.
%}

%% Options

isTraining = true; % Set to false if you already trained the network

%% Load data

[XMerged, YMerged] = mergeData(60000);

XTrain = XMerged(1:80);
YTrain = YMerged(1:80);
XTest = XMerged(81:104);
YTest = YMerged(81:104);

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

if isTraining
    numFeatures = 1;
    numHiddenUnits = 200;
    numClasses = 2;
    
    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 2, ...
        'Verbose', 0, ...
        'Plots', 'training-progress', ...
        'MiniBatchSize', 4);
    
    net = trainNetwork(XTrain, YTrain, layers, options);
    save('net', 'net')
else
    load('net.mat')
end

%% Test

X = XTest{5};
Y = YTest{5};
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
ylabel('Activity')
title('Predicted Activities')
legend('Predicted', 'Test data')

ax2 = nexttile;
for i = 1:numel(classes)
    label = classes(i);
    idx = find(Y == label);
    plotVal = NaN(size(X));
    plotVal(idx) = X(idx);
    hold on
    plot(plotVal)
end

xlabel('Time (ms)')
ylabel('Position (deg)')
title('Example')
legend(classes)
linkaxes([ax1, ax2], 'x')

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