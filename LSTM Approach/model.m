%{
Code by Kellen Vu

This program uses an LSTM network to identify saccades vs. non-saccades.
%}

%% Options

isTraining = false; % Set to false if you already trained the network

%% Load data

[XMerged, LMerged] = mergeData(1000);

mTrain = round(size(XMerged, 1) * 0.8);

XTrain = XMerged(1:mTrain);
LTrain = LMerged(1:mTrain);
XTest = XMerged(mTrain:end);
LTest = LMerged(mTrain:end);

% Plot one example
%{
X = XTrain{1};
classes = categories(LTrain{1});

figure
for i = 1:numel(classes)
    label = classes(i);
    idx = find(LTrain{1} == label);
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
    
    [net, info] = trainNetwork(XTrain, LTrain, layers, options);
    save('net.mat', 'net', 'info')
else
    load('net.mat')
end

%% Test Plot

classes = categories(LTest{1});
LPred = classify(net, XTest);

% Plot some examples
XTestPlot = cat(2, XTest{1:100});
LTestPlot = cat(2, LTest{1:100});
LPredPlot = cat(2, LPred{1:100});
figure
tiledlayout(2, 1)
ax1 = nexttile;
hold on
plot(LTestPlot, '.-')
plot(LPredPlot)

xlabel('Time step')
title('Predicted Classes')
legend('Ground truth', 'Predicted')

ax2 = nexttile;
hold on
for i = 1:numel(classes)
    label = classes{i};
    idx = find(LTestPlot == label);
    plotVal = NaN(size(XTestPlot));
    plotVal(idx) = XTestPlot(idx);
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

XTestEval = cat(2, XTest{1:end});
LTestEval = cat(2, LTest{1:end});
LPredEval = cat(2, LPred{1:end});

kappa = cohensKappa(LTestEval, LPredEval);
fprintf("Cohens Kappa: %d\n", kappa)

%{
truePos = 0;
actualPos = 0;
N = 400;
for i = 1:9
    X = cat(2, XTrain{i * 400 + 1:(i + 1) * 400});
    YPred = classify(net, X, 'MiniBatchSize', miniBatchSize);
    Y = cat(2, LTrain{i * 400 + 1:(i + 1) * 400});
    truePos = truePos + sum((YPred == Y) & (Y == 'Saccade'));
    actualPos = actualPos + sum(Y == 'Saccade');
end

fprintf('Train Set: ClassifiedAsSaccadeByNN / ClassifiedAsSaccadeByThreshold = %d\n', truePos / actualPos)
%}

beep

%% Functions

function [XMerged, LMerged] = mergeData(exampleSize)
    % Merge multiple data.mat files into one X cell array and one L cell array.
    % :param exampleSize: The maximum length of each example
    % :return XMerged: The merged input data (cell array)
    % :return LMerged: The merged label data (cell array)
    XMerged = {};
    LMerged = {};
    
    % Load .mat files
    dataFiles = dir('data/*_data.mat');
    for i = 1:length(dataFiles)
        file = fullfile(dataFiles(i).folder, dataFiles(i).name);
        load(file, 'X', 'L')
        L = categorical(L, [0, 1], {'Non-saccade', 'Saccade'});
        
        % Split each file into multiple examples (to make it easier on the GPU)
        N = floor(numel(X) / exampleSize);
        X = reshape(X(1:N * exampleSize), exampleSize, [])';
        X = num2cell(X, 2);
        L = reshape(L(1:N * exampleSize), exampleSize, [])';
        L = num2cell(L, 2);

        XMerged = [XMerged; X];
        LMerged = [LMerged; L];
    end
end