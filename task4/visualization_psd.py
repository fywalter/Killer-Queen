import numpy as np
import matplotlib.pyplot as plt
import os
#from scipy import signal
#from biosppy.signals import ecg
#from biosppy.plotting import plot_ecg
#from hrvanalysis import get_time_domain_features
#import math
#import csv

def log_psd(arr):
    mag = np.linalg.norm(arr, axis=2)
    log_psd = 20 * np.log10(mag)
    return log_psd


def main():
    print()
    print('***************By Killer Queen***************')

    train_all = np.load('./data/train_persub_x.npy')
    train_labels = np.load('./data/train_labels.npy')
    train_eeg1 = train_all[:,:,:,0]
    train_eeg2 = train_all[:,:,:,1]
    train_emg = train_all[:,:,:,2]
    eeg1_sub1 = train_eeg1[:21600,:,:]
    eeg1_sub2 = train_eeg1[21600:21600*2,:,:]
    eeg1_sub3 = train_eeg1[21600*2:,:,:]
    eeg2_sub1 = train_eeg2[:21600,:,:]
    eeg2_sub2 = train_eeg2[21600:21600*2,:,:]
    eeg2_sub3 = train_eeg2[21600*2,:,:]
    y_sub1 = train_labels[:21600]
    y_sub2 = train_labels[21600:21600*2]
    y_sub3 = train_labels[21600*2:]
    idx_sub1_class1 = np.where(y_sub1==1)[0]
    idx_sub1_class2 = np.where(y_sub1==2)[0]
    idx_sub1_class3 = np.where(y_sub1==3)[0]
    idx_sub2_class1 = np.where(y_sub2==1)[0]
    idx_sub2_class2 = np.where(y_sub2==2)[0]
    idx_sub2_class3 = np.where(y_sub2==3)[0]
    idx_sub3_class1 = np.where(y_sub3==1)[0]
    idx_sub3_class2 = np.where(y_sub3==2)[0]
    idx_sub3_class3 = np.where(y_sub3==3)[0]
    
    log_psd_eeg1_sub1_class1 = log_psd(eeg1_sub1[idx_sub1_class1])
    
    if not os.path.exists('./visualization_psd'):
        os.mkdir('./visualization_psd')

    # train_eeg1, train_eeg2, train_emg, y_train = shuffle(train_eeg1, train_eeg2, train_emg, y_train)
    # plot_some_imgs(train_eeg1, train_eeg2, train_emg, y_train)
    # plot_full_length_img(train_eeg1, train_eeg2, train_emg, y_train)
    # print_statistics(train_eeg1, train_eeg2, train_emg, y_train)
    # plot_fft(train_eeg1, train_eeg2, train_emg, y_train)


main()
