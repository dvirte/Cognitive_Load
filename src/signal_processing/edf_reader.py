from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from EDFFile import EDFFile
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Create an EDFFile object by providing the path to your EDF file
edf_file = EDFFile('data/participant_01/S01/01.edf')

# # Plot the ICA heatmap
# edf_file.plot_ica_heatmap()
#
# # Plot the raw signal data
# edf_file.plot_signal()

# Step 1: Split events into train and test
train_events, test_events = edf_file.split_events()

# Step 2: Label windows
window_labels, window_split = edf_file.label_windows(train_events, test_events)


# Step 3: Extract train and test data
train_indices = np.where(window_split == 1)[0]
test_indices = np.where(window_split == 0)[0]

X_train = edf_file.extract_features(train_indices)
X_test = edf_file.extract_features(test_indices)

Y_train = window_labels[train_indices]
Y_test = window_labels[test_indices]

# Remove the 0 labels
Y_test_new = Y_test[Y_test != 0]
X_test_new = X_test[Y_test != 0]

# Initialize the SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the SVM on extracted features
svm_model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = svm_model.predict(X_test)

Y_pred_new = svm_model.predict(X_test_new)

# Evaluate performance
print("Classification Report - including 0:")
print(classification_report(Y_test, Y_pred))

print("Confusion Matrix - including 0:")
print(confusion_matrix(Y_test, Y_pred))

print("Classification Report - excluding 0:")
print(classification_report(Y_test_new, Y_pred_new))

print("Confusion Matrix - excluding 0:")
print(confusion_matrix(Y_test_new, Y_pred_new))

# Lazy Predict
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test_new, Y_train, Y_test_new)
print(models)


# Plot the jaw features statistics
edf_file.jaw_features()

# list_of_stat = ['MAV_CD4', 'STD_CD4', 'MAV_CD5', 'STD_CD5', 'MAV_x']