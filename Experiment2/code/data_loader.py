import scipy.io as sio
import torch
import numpy as np

def load_data(subject_id, session):
 # Loading data for sessions 1 and 2
    mat_data_train1 = sio.loadmat(f'path to session 1 mat file')
    mat_data_train2 = sio.loadmat(f'path to session 2 mat file')
    
    data_train1 = mat_data_train1['data']
    labels_train1 = mat_data_train1['labels'].reshape(-1)
    data_train1 = data_train1 / np.max(data_train1, axis=2, keepdims=True)
    
    data_train2 = mat_data_train2['data']
    labels_train2 = mat_data_train2['labels'].reshape(-1)
    data_train2 = data_train2 / np.max(data_train2, axis=2, keepdims=True)

    
 # Combining data from sessions 1 and 2
    X_all = torch.tensor(np.concatenate((data_train1, data_train2), axis=0)).float()
    y_all = torch.tensor(np.concatenate((labels_train1, labels_train2), axis=0) - 1, dtype=torch.long)
    
    # 60-40 split
    train_size = int(0.6 * len(X))
    indices = list(range(len(X)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test

def load_test_data(subject_id, session):
    mat_data_test = sio.loadmat(f'path to .mat file')
    data_test = mat_data_test['data'] / np.max(mat_data_test['data'], axis=2, keepdims=True)
    labels_test = mat_data_test['labels'].reshape(-1)
    
    X_test = torch.tensor(data_test).float()
    y_test = torch.tensor(labels_test - 1, dtype=torch.long)
    
    return X_test, y_test
