clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results

load("predictionsRF")

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

% Compute Feature Importance
importance = randomForestModel.predictorImportance(); % Use the trained model object

% Visualize Feature Importance
bar(importance);
xlabel('Feature Index');
ylabel('Importance Score');
title('Feature Importance');