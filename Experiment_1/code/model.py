
'''
This is the code for the proposed hybrid-model EEG_CNN-GRU
Author- Anam Suri
'''
import torch
import toch.nn as nn
import torch.nn.functional as F



class EEGCNN_GRU(nn.Module):
    def __init__(self, input_channels, sequence_length, hidden_size, num_classes):
        super(EEGCNN_GRU, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(2)
        self.gru = nn.GRU(64, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = F.elu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
