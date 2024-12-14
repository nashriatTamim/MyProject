clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results

load("predictionsLogistic")


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

