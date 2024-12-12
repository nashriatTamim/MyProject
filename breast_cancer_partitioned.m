%% Load data
file_path = 'breast_cancer_selected_features_2.csv';
data = readtable(file_path);


%% Display the data
disp(head(data));

%% Separate target variable and features

x=data{:,setdiff(data.Properties.VariableNames, 'Diagnosis')};
y=data.Diagnosis;

%% Partition the data into training and testing sets
partitiondata = cvpartition(y, 'HoldOut', 0.2);
X_train = data(training(partitiondata), :);
X_test = data(test(partitiondata), :);

%% Save the training and test data
writetable(X_train,'Partitionedtrainset2.csv');
writetable(X_test,'Partitionedtestset2.csv');
