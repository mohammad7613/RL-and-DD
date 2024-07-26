# # # # # Step 0: Import Required Modules # # # # #

import numpy as np
import matplotlib.pyplot as plt

from ConMod import *

import os
import h5py

import pandas as pd

# # # # # Step 1: Import and Prepare Required Data (Subject EEG Data and Subjects List)  # # # # #

BehavioralData = pd.read_excel(r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx')

with h5py.File(r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat', 'r') as f:

    raw_data = f['All_data_All'][:]


with h5py.File(r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat', 'r') as f:

    data_lengths = f['data_lengths'][:]

# folder path
dir_path = r'E:\\HWs\Msc\\Research\\Research\\Depression Dataset\\Testing Preprocess'

# list to store files
subjects = []

# Iterate directory
for path in os.listdir(dir_path):

    subjects.append(path)

subjects_of_interest = []                        
            
for i in range(len(subjects)):

    idx = np.where((BehavioralData['id'][:]) == int(subjects[i]))[0][0]
    subjects_of_interest.append(idx)

raw_data = raw_data.transpose(3, 1, 2, 0)

# # # # # Step 3: Select a Subject and Data to be Analysed # # # # #

subject = 10

Data_s = raw_data[subject, 1 : int(data_lengths[0, subject]), :, :]
print(Data_s.shape)

Conn_Data_1_s = np.squeeze(np.mean(Data_s[:20, :, :], axis = 0))
Conn_Data_2_s = np.squeeze(np.mean(Data_s[-20:, :, :], axis = 0))

channels_n = np.arange(0, 14)
channels = ['PF', 'LF', 'RF', 'MFC', 'LT', 'RT', 'LFC', 'RFC', 'MPC', 'LPC', 'RPC', 'MP', 'LPO', 'RPO']


# # # # # Step 4: Calculating the Orders Matrix # # # # #

output_filename = r'D:\\AIRLab_Research\\Connectivity\\GrangerCausality\\OrdersMat\\' + 'All_FB_' + str(BehavioralData['id'][subjects_of_interest[subject]]) + '_ChanNum_' + str(len(channels_n))

if os.path.exists(output_filename):

    orders_mat_s = np.load(output_filename + '.npy')

else:

    orders_mat_s = OrderEstimate_byChannels((Conn_Data_1_s + Conn_Data_2_s) / 2, channels_n, 200, 10, 4)
    np.save(output_filename, orders_mat_s)

# # # # # Step 5: Calculating the Granger Causality Values # # # # #

window_length = 200
overlap_ratio = 0.99

output_filename = r'D:\\AIRLab_Research\\Connectivity\\GrangerCausality\\GC_values\\' + 'All_FB_Phase_1_' + str(BehavioralData['id'][subjects_of_interest[subject]]) + '_ChanNum_' + str(len(channels_n)) + '_OverlapRatio_' + str(overlap_ratio) + '_WinLen_' + str(window_length)

if os.path.exists(output_filename):

    GC_vals_1_s = np.load(output_filename + '.npy')

else:

    GC_vals_1_s = GrangerCausalityEstimator(Conn_Data_1_s, channels_n, window_length, overlap_ratio, orders_mat_s)
    np.save(output_filename, GC_vals_1_s)

output_filename = r'D:\\AIRLab_Research\\Connectivity\\GrangerCausality\\GC_values\\' + 'All_FB_Phase_2_' + str(BehavioralData['id'][subjects_of_interest[subject]]) + '_ChanNum_' + str(len(channels_n)) + '_OverlapRatio_' + str(overlap_ratio) + '_WinLen_' + str(window_length)

if os.path.exists(output_filename):

    GC_vals_2_s = np.load(output_filename + '.npy')

else:

    GC_vals_2_s = GrangerCausalityEstimator(Conn_Data_2_s, channels_n, window_length, overlap_ratio, orders_mat_s)
    np.save(output_filename, GC_vals_2_s)

# # # # # Step 6: Visulizing a sample of GC Dynamics # # # # #

transmitter_channel = 8

t_ = np.arange(-0.4, 1.2, 1.6 / len(GC_vals_1_s[:, 0, 0]))
print(len(GC_vals_1_s[:, 0, 0]))

plt.figure(figsize = (15, 8))
for receiver_channel in range(14):

    if transmitter_channel != receiver_channel:

        plt.plot(t_, GC_vals_1_s[:, receiver_channel, transmitter_channel], label = str(channels[receiver_channel]))

plt.legend()
plt.title("Granger Causality Scores Transmitting from Channel " + str(channels[transmitter_channel]) + " In the first Block of Training")
plt.show()

plt.figure(figsize = (15, 8))
for receiver_channel in range(14):

    if transmitter_channel != receiver_channel:

        plt.plot(t_, GC_vals_2_s[:, receiver_channel, transmitter_channel], label = str(channels[receiver_channel]))

plt.legend()
plt.title("Granger Causality Scores Transmitting from Channel " + str(channels[transmitter_channel]) + " In the first Block of Training")
plt.show()

