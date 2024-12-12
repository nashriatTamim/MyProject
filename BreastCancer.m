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

% Train Logistic Regression Model
logisticModel = fitglm(X_Train, Y_Train, 'Distribution', 'binomial'); % Logistic Regression

% Predict using Logistic Regression
predictionsLogistic = predict(logisticModel, X_Test);
predictionsLogistic = round(predictionsLogistic); % Convert probabilities to 0 or 1

% Evaluate Logistic Regression Model
accuracyLogistic = sum(predictionsLogistic == Y_Test) / length(Y_Test) * 100;
disp(['Logistic Regression Accuracy: ', num2str(accuracyLogistic), '%']);

% Confusion Matrices
disp('Confusion Matrix for Logistic Regression:');
confusionchart(Y_Test, predictionsLogistic);

% Compute confusion matrix
confusionMatrix = confusionmat(Y_Test, predictionsLogistic);

% Extract values from confusion matrix
TP = confusionMatrix(2, 2); % True Positives
TN = confusionMatrix(1, 1); % True Negatives
FP = confusionMatrix(1, 2); % False Positives
FN = confusionMatrix(2, 1); % False Negatives

% Calculate performance metrics
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP); % Handle division by zero
recall = TP / (TP + FN); % Sensitivity
F1_score = 2 * (precision * recall) / (precision + recall); % Harmonic mean

% Display the results
fprintf('Accuracy LR: %.2f%%\n', accuracy * 100);
fprintf('Precision LR: %.2f\n', precision);
fprintf('Recall LR: %.2f\n', recall);
fprintf('F1-Score LR: %.2f\n', F1_score);

% Assuming Y_Test (true labels) and Y_Prob (predicted probabilities)

[X, Y, T, AUC] = perfcurve(Y_Test, predictionsLogistic, 1); % '1' indicates the positive class

% Plot the ROC curve
figure;
plot(X, Y, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);
grid on;

% Train Random Forest Model
randomForestModel = TreeBagger(100, X_Train, Y_Train, 'Method', 'classification', 'OOBPrediction', 'on');

% Predict using Random Forest
predictionsRF = str2double(predict(randomForestModel, X_Test)); % Convert predictions to numeric

% Evaluate Random Forest Model
accuracyRF = sum(predictionsRF == Y_Test) / length(Y_Test) * 100;
disp(['Random Forest Accuracy: ', num2str(accuracyRF), '%']);

disp('Confusion Matrix for Random Forest:');
confusionchart(Y_Test, predictionsRF);

confusionMatrix = confusionmat(Y_Test, predictionsRF);

% Extract values from confusion matrix
TP = confusionMatrix(2, 2); % True Positives
TN = confusionMatrix(1, 1); % True Negatives
FP = confusionMatrix(1, 2); % False Positives
FN = confusionMatrix(2, 1); % False Negatives

% Calculate performance metrics
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP); % Handle division by zero
recall = TP / (TP + FN); % Sensitivity
F1_score = 2 * (precision * recall) / (precision + recall); % Harmonic mean

% Display the results
fprintf('Accuracy RF: %.2f%%\n', accuracy * 100);
fprintf('Precision RF: %.2f\n', precision);
fprintf('Recall RF: %.2f\n', recall);
fprintf('F1-Score RF: %.2f\n', F1_score);



