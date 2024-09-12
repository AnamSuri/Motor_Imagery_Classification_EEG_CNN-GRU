''''
Author: Anam Suri
This is the main file for Experiment 3 (Cross-subject classification)
User can import the relevant files and libraries for the program to compile.

''''


import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, roc_curve, auc
import seaborn as sns
from scipy.optimize import brentq
from scipy.interpolate import interp1d



seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

training_subjects = [2, 6, 7, 12, 13, 19, 20, 21, 22, 23]
testing_subjects = [i for i in range(1, 26) if i not in training_subjects]

input_channels = 32
sequence_length = 1000
hidden_size = 128
num_classes = 2
batch_size = 8
num_epochs = 40
num_folds = 10
best_val_loss = float('inf')
best_val_accuracy = 0.0
model = EEGCNN_GRU(input_channels, sequence_length, hidden_size, num_classes).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

X_train, y_train = load_and_combine_data(training_subjects)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_train, y_train, test_size=0.4, random_state=42, stratify=y_train)

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    val_acc_fold = []
    val_loss_fold = []
    X_train_fold = X_train[train_index]
    y_train_fold = y_train[train_index]
    X_val_fold = X_train[val_index]
    y_val_fold = y_train[val_index]
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader_fold, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader_fold, criterion, device)
        val_acc_fold.append(val_accuracy)
        val_loss_fold.append(val_loss)
    if best_val_loss > np.mean(val_loss_fold) and best_val_accuracy < np.mean(val_acc_fold):
        best_val_loss = np.mean(val_loss_fold)
        best_val_accuracy = np.mean(val_acc_fold)
        best_model_state_dict = model.state_dict()

# Loading best model state dict
model.load_state_dict(best_model_state_dict)
torch.save(model.state_dict(), 'cross_subjects_model.pth')

# Testing on 40% test data
test_dataset_full = TensorDataset(X_test_full, y_test_full)
test_loader_full = DataLoader(test_dataset_full, batch_size=batch_size, shuffle=False)
test_accuracy, true_labels, predicted_labels, predicted_probs_np = test(model, test_loader_full, device)
print(f"Test accuracy on 40% test data: {test_accuracy:.2f}%")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
precision = precision_score(true_labels, predicted_labels)
print("precsion of 40 % test data: ", precision)
recall = recall_score(true_labels, predicted_labels)
print("recall of 40 % test data: ", recall)
f1 = f1_score(true_labels, predicted_labels)
print("f1 of 40 % test data: ", f1)

# Cross-subject
for test_subject_id in testing_subjects:
    print(f"Testing with Subject {test_subject_id}")
    
    # Load test data for all 5 sessions
    all_test_data = []
    all_test_labels = []
    for session in range(1, 6):
        mat_data_test = sio.loadmat(f'path to mat file')
        data_test = mat_data_test['data']
        labels_test = mat_data_test['labels'].reshape(-1)
        data_test = data_test / np.max(data_test, axis=2, keepdims=True)  # Normalize data
        all_test_data.append(data_test)
        all_test_labels.append(labels_test)
    all_test_data = np.concatenate(all_test_data, axis=0)
    all_test_labels = np.concatenate(all_test_labels)
    
    X_test = torch.tensor(all_test_data).float()
    y_test = torch.tensor(all_test_labels - 1, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_accuracy, true_labels, predicted_labels, predicted_probs_np = test(model, test_loader, device)
    print(f"Test accuracy for subject {test_subject_id}: {test_accuracy:.2f}%")
    
    # Calculate additional metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs_np[:, 1])
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_score = auc(fpr, tpr)

    # Save results
    results = {
        'test_subject_id': test_subject_id,
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_val_loss,
        'test_accuracy': test_accuracy,
        'TPR': TPR,
        'FPR': FPR,
        'FNR': FNR,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'EER': eer,
        'AUC': auc_score
    }
    results_df = pd.DataFrame([results])
    #results_df.to_csv(f'results_test_sub-{test_subject_id}.csv', index=False)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Test Subject {test_subject_id}')
    plt.savefig(f'confusion_matrix_test_sub-{test_subject_id}.png')
    plt.close()

print("Training and testing completed for all subjects.")

