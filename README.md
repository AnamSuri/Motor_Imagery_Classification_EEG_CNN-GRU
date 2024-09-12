# Motor_Imagery_Classification_EEG_CNN-GRU
This repo describes an implementation of the models described in "Exploring BCI Literacy in Motor Imagery Task Classification Performance".
The repository consists of experiments and code files for Motor Imagery Classification on "A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface.
In this, we have proposed a novel hybrid model EEG_CNN-GRU consisting of Convolutional Neural Networks (CNNs) and Gated Recurrent Units (GRU) to capture spatio-temporal patterns in the EEG data. The architecture of the proposed model is gievn below:
![Alt text](eeg_cnn_gru_architecture.png)

We have performed different cross-session and cross-subject experiments. The genral scenario of the experiments performed in the literature is described in the follwoing figure. For this study, we have incorporated scenarios 'c' and 'd' only.
![Alt text](experimental_scenarios.jpg)

## Requirements
In order to run this code you need to install the following modules:

Numpy and Scipy (http://www.scipy.org/install.html)

Scikit-Learn (http://scikit-learn.org/stable/install.html)

Python 3.7
Pytroch 1.3.1
Cudatoolkit 10.1.243
Cudnn 7.6.3
  



