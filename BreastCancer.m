%% Comparison of Random Forests and Logistic Regression on Wisconsin Breast Cancer Dataset (Diagnostic)
 
%% Initialisation
clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results

%% Load Partitioned data
train_Data = readtable("Partitionedtrainset2.csv");

head(train_Data)
train_Data.Properties.VariableNames

train_Data = table2array(train_Data); % Converts table to matrix
%test_Data = table2array(test_Data);

% Split features (X) and target variable (Y)
X_Train = train_Data(:, 1:14);  % Features from training data
Y_Train = train_Data (:,15);
%X_Test = test_Data (:,1:14);
%Y_Test = test_Data (:,15);

% Train Logistic Regression Model
logisticModel = fitglm(X_Train, Y_Train, 'Distribution', 'binomial'); % Logistic Regression

% Predict using Logistic Regression
%predictionsLogistic = predict(logisticModel, X_Test);
%predictionsLogistic = round(predictionsLogistic); % Convert probabilities to 0 or 1

save('logisticModel')

coefficients = logisticModel.Coefficients.Estimate;
featureNames = {'concave_points_worst', 'perimeter_worst', 'concave_points_mean', 'radius_worst', 'perimeter_mean','area_worst', 'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst', 'compactness_worst', 'radius_se', 'perimeter_se', 'area_se'}; % Update as necessary
figure;
bar(coefficients(2:end)); % Skip intercept
set(gca, 'XTickLabel', featureNames, 'XTick', 1:numel(featureNames), 'XTickLabelRotation', 45);
xlabel('Features');
ylabel('Coefficient Value');
title('Feature Importance (Logistic Regression)');
grid on;

%% Manual K-Fold Cross-Validation for Logistic Regression
k = 5; % Number of folds
cv = cvpartition(Y_Train, 'KFold', k); % Create cross-validation partitions

accuracy = zeros(k, 1); % To store accuracy for each fold

for i = 1:k
    % Get training and validation indices for the current fold
    trainIdx = training(cv, i);
    valIdx = test(cv, i);
    
    % Split data into training and validation sets
    X_TrainFold = X_Train(trainIdx, :);
    Y_TrainFold = Y_Train(trainIdx);
    X_ValFold = X_Train(valIdx, :);
    Y_ValFold = Y_Train(valIdx);
    
    % Train logistic regression model
    logisticModel = fitglm(X_TrainFold, Y_TrainFold, 'Distribution', 'binomial');
    
    % Predict on validation set
    predictions = predict(logisticModel, X_ValFold);
    predictions = round(predictions); % Convert probabilities to class labels (0 or 1)
    
    % Compute accuracy for this fold
    accuracy(i) = sum(predictions == Y_ValFold) / numel(Y_ValFold) * 100;
end

% Compute and display overall cross-validation accuracy
meanAccuracy = mean(accuracy);
fprintf('Mean Cross-Validation Accuracy: %.2f%%\n', meanAccuracy);

figure;
bar(1:k, accuracy, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k'); % Create bar graph
xlabel('Fold Number');
ylabel('Accuracy (%)');
title('Cross-Validation Accuracy');
grid on;
xticks(1:k); % Ensure ticks are at each fold
ylim([min(accuracy) - 5, 100]); % Adjust y-axis for better visualization

% Annotate each bar with its accuracy value
for i = 1:k
    text(i, accuracy(i) + 1, sprintf('%.2f%%', accuracy(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
