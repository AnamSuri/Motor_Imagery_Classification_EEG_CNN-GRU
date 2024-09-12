import scipy.io as sio
import torch
import numpy as np

def load_data(subject_id, session):
    mat_data_train = sio.loadmat(f'path_to_.mat file')
    data = mat_data_train['data'] / np.max(mat_data_train['data'], axis=2, keepdims=True)
    labels = mat_data_train['labels'].reshape(-1)
    
    X = torch.tensor(data).float()
    y = torch.tensor(labels - 1, dtype=torch.long)
    
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
