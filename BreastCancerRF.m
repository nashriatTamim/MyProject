%% Comparison of Random Forests and Logistic Regression on Wisconsin Breast Cancer Dataset (Diagnostic)
 
%% Initialisation
clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results

%% Load Partitioned data
train_Data = readtable("Partitionedtrainset2.csv");
test_Data = readtable("Partitionedtestset2.csv");

head(train_Data)
train_Data.Properties.VariableNames

train_Data = table2array(train_Data); % Converts table to matrix
test_Data = table2array(test_Data);

% Split features (X) and target variable (Y)
X_Train = train_Data(:, 1:14);  % Features from training data
Y_Train = train_Data (:,15);
X_Test = test_Data (:,1:14);
Y_Test = test_Data (:,15);

% Train Random Forest Model
randomForestModel = TreeBagger(100, X_Train, Y_Train, 'Method', 'classification', 'OOBPrediction', 'on','OOBPredictorImportance','on');



% Predict using Random Forest
predictionsRF = str2double(predict(randomForestModel, X_Test)); % Convert predictions to numeric

save('predictionsRF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nTrees= 55;% number of trees

k=5; %number of folds for validation

% Create a partition for k-fold cross-validation
cv = cvpartition(Y_Train, 'KFold', k);

% Initialize an array to store the accuracy for each fold
accuracy = zeros(k, 1);


% Perform cross-validation
for i = 1:k
    % Get the training and validation indices for this fold
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    % Training and testing indices for this fold
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    % Train model on training fold
    model = TreeBagger(100, X_Train(trainIdx, :), Y_Train(trainIdx), 'Method', 'classification');

    % Test model on test fold
    predictions = str2double(predict(model, X_Train(testIdx, :)));
    
    % Compute accuracy for this fold
    accuracy(i) = sum(predictions == Y_Train(testIdx)) / numel(Y_Train(testIdx));
end
% Calculate the average accuracy across all folds
averageAccuracy = mean(accuracy);
fprintf('Average Cross-Validation Accuracy: %.2f%%\n', averageAccuracy * 100);

% Assuming `accuracy` contains accuracy for each fold
folds = 1:cv.NumTestSets; % Number of folds
bar(folds, accuracy * 100); % Convert to percentage
xlabel('Fold Number');
ylabel('Accuracy (%)');
title('Accuracy per Fold');
ylim([0 100]); % Set y-axis limits to 0-100%
grid on;

% Annotate each bar with its accuracy value
for i = 1:numel(folds)
    text(folds(i), accuracy(i) * 100 + 2, sprintf('%.2f%%', accuracy(i) * 100), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

featureNames = {'concave_points_worst', 'perimeter_worst', 'concave_points_mean', 'radius_worst', 'perimeter_mean','area_worst', 'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst', 'compactness_worst', 'radius_se', 'perimeter_se', 'area_se'}; % Update as necessary

    % Compute feature importance
    importance = randomForestModel.OOBPermutedPredictorDeltaError;

    % Visualize feature importance
    figure;
    bar(importance);
    set(gca, 'XTickLabel', featureNames, 'XTick', 1:numel(featureNames), 'XTickLabelRotation', 45);
    xlabel('Features');
    ylabel('Importance');
    title('Feature Importance (Random Forest)');
    grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%