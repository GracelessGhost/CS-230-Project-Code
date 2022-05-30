#!/usr/bin/env python3
import numpy as np
import uneye
import scipy.io as io
from os.path import join as pj
import matplotlib.pyplot as plt


# Parameters
min_sacc_dur = 6  # Minimum saccade duration in ms
min_sacc_dist = 10  # Minimum saccade distance in ms
sampfreq = 1000  # Hz
weights_name = 'weights_synthetic'


# Data
datapath = 'data/dataset1'  # Folder with example data
x_filename = 'dataset1_1000hz_X_setA.csv'
y_filename = 'dataset1_1000hz_Y_setA.csv'
labels_filename = 'dataset1_1000hz_Labels_setA.csv'



def main():
    """
    Xtrain = np.loadtxt(pj(datapath, x_filename), delimiter=',')
    Ltrain = np.loadtxt(pj(datapath, labels_filename), delimiter=',')
    fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True)
    for i in range(5):
        X = Xtrain[i + 5, :]
        L = Ltrain[i + 5, :]
        saccades = np.empty(X.shape)
        saccades[:] = np.nan
        saccades[L == 1] = X[L == 1]
        ax[i].plot(X)
        ax[i].plot(saccades)
    plt.show()
    """
    # Parameters
    sampfreq = 1000  # Hz
    weights_name = 'weights_synthetic'
    min_sacc_dur = 6  # In ms
    min_sacc_dist = 10  # In ms

    # Data
    datapath = ''
    x_filename = 'X.mat'
    y_filename = 'Y.mat'

    # Load data
    Xtest = io.loadmat(datapath + 'X.mat')['X']
    Ytest = io.loadmat(datapath + 'Y.mat')['Y']

    # Prediction
    model = uneye.DNN(
        weights_name=weights_name,
        sampfreq=sampfreq,
        min_sacc_dur=min_sacc_dur,
        min_sacc_dist=min_sacc_dist)
    Prediction, Probability = model.predict(Xtest, Ytest)
    print('done')

    # Plot
    i = 0
    fig, ax = plt.subplots(nrows=2, sharex=True)
    plt.suptitle('Example')
    x_trace = Xtest[i, :] - np.min(Xtest[i, :])
    ax[0].plot(x_trace, c=[0, 0.5, 0.8])
    ax[0].set_ylabel('Eye pos')

    saccades = np.empty(x_trace.shape)
    saccades[:] = np.nan
    saccades[Prediction == 1] = x_trace[Prediction == 1]
    ax[0].plot(saccades)

    ax[1].plot(Prediction[i, :], label='Binary prediction', c=[0, 0.6, 0.3])
    ax[1].set_xlabel('Time')
    plt.show()


if __name__ == "__main__":
    main()
