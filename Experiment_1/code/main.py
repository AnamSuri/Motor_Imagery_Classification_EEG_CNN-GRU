'''
Author: Anam Suri
This is the main file for Experiment 1.
Here the user can set path to their data and rest will work fine.

'''

# importing necessary libraries
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from model import EEGCNN_GRU
from train_test_functions import train, validate, test
from data_loader import load_data, load_test_data
from results import save_results, plot_confusion_matrix

## Setting random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## Main loop to iterate through 25 subjects
results_list = []
for subject_id in range(1, 26):
    print(f"Processing Subject {subject_id}")

    ## Load session 1 data
    X_train, y_train, X_test, y_test = load_data(subject_id, session=1)

    # Initialize model, criterion, optimizer
    input_channels, sequence_length, hidden_size, num_classes = 32, 1000, 128, 2
    batch_size, num_epochs, num_folds = 32, 40, 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGCNN_GRU(input_channels, sequence_length, hidden_size, num_classes).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    ## Training and cross-validation
    best_val_loss, best_val_accuracy = float('inf'), 0.0
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        train_loader, val_loader = create_dataloaders(X_train, y_train, train_index, val_index, batch_size)
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        if val_loss < best_val_loss and val_accuracy > best_val_accuracy:
            best_val_loss, best_val_accuracy = val_loss, val_accuracy
            best_model_state_dict = model.state_dict()

    ## Load best model
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(), f'subject_{subject_id}_model.pth')

    ## Test with session 1 data
    test_loader = create_test_loader(X_test, y_test, batch_size)
    test_accuracy, true_labels, predicted_labels, _ = test(model, test_loader, device)
    
    ## Additional metrics
    precision_sess1 = precision_score(true_labels, predicted_labels, average='weighted')
    recall_sess1 = recall_score(true_labels, predicted_labels, average='weighted')
    f1_sess1 = f1_score(true_labels, predicted_labels, average='weighted')
    confusion_matrix_sess1 = confusion_matrix(true_labels, predicted_labels)

    # Testing with session 5 data
    X_test5, y_test5 = load_test_data(subject_id, session=5)
    test_loader_5 = create_test_loader(X_test5, y_test5, batch_size)
    test_accuracy_5, true_labels_5, predicted_labels_5, predicted_probs_np_5 = test(model, test_loader_5, device)

    # Calculate session 5 metrics and plot confusion matrix
    save_results(subject_id, best_val_loss, best_val_accuracy, test_accuracy, precision_sess1, recall_sess1, f1_sess1, confusion_matrix_sess1,
                 test_accuracy_5, precision_sess5, recall_sess5, f1_sess5, confusion_matrix_sess5)
    plot_confusion_matrix(subject_id, confusion_matrix_sess5)

print("Training and testing completed for all subjects.")

