
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:27:56 2020

@author: venusroshdi
"""

import numpy as np
from statsmodels.stats import weightstats

sensor = 'Back'
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\' + sensor + '\\'

accel_statistics = np.load(filepath+sensor+'accel_statistics.npy')
gyro_statistics = np.load(filepath+sensor+'gyro_statistics.npy')
bio = np.load(filepath+sensor+'bio.npy')
Y = np.load(filepath+sensor+'Y.npy')
subject_number = np.load(filepath+sensor+'subject_number.npy')

F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_bio, nonF_bio = [], [], [], [], [], []
F_Y, nonF_Y, F_subject_number, nonF_subject_number = [], [], [], []
for i in range(len(Y)):
    if Y[i]==1:
        F_accel_statistics.append(accel_statistics[i:i+1,:])
        F_gyro_statistics.append(gyro_statistics[i:i+1,:])
        F_bio.append(bio[i:i+1,:])
        F_Y.append(Y[i:i+1])
        F_subject_number.append(subject_number[i:i+1,:])
    if Y[i]==0:
        nonF_accel_statistics.append(accel_statistics[i:i+1,:])
        nonF_gyro_statistics.append(gyro_statistics[i:i+1,:])
        nonF_bio.append(bio[i:i+1,:])
        nonF_Y.append(Y[i:i+1])
        nonF_subject_number.append(subject_number[i:i+1,:])        
F_accel_statistics = np.concatenate(F_accel_statistics, axis=0)
F_gyro_statistics = np.concatenate(F_gyro_statistics, axis=0)
F_bio = np.concatenate(F_bio, axis=0)
F_Y = np.concatenate(F_Y, axis=0)
F_subject_number = np.concatenate(F_subject_number, axis=0)
nonF_accel_statistics = np.concatenate(nonF_accel_statistics, axis=0)
nonF_gyro_statistics = np.concatenate(nonF_gyro_statistics, axis=0)
nonF_bio = np.concatenate(nonF_bio, axis=0)
nonF_Y = np.concatenate(nonF_Y, axis=0)
nonF_subject_number = np.concatenate(nonF_subject_number, axis=0)

F_age, F_gender = F_bio[:,1], F_bio[:,0]
nonF_age, nonF_gender = nonF_bio[:,1], nonF_bio[:,0]

            ##################
RMS_bio = np.load(filepath+sensor+'RMS_bio.npy')
RMS_Y = np.load(filepath+sensor+'RMS_Y.npy')
RMS_accel_statistics = np.load(filepath+sensor+'RMS_accel_statistics.npy')
RMS_gyro_statistics= np.load(filepath+sensor+'RMS_gyro_statistics.npy')

F_RMS_accel_statistics, nonF_RMS_accel_statistics, F_RMS_gyro_statistics, nonF_RMS_gyro_statistics, F_RMS_bio, nonF_RMS_bio = [], [], [], [], [], []
F_RMS_Y, nonF_RMS_Y = [], []
for i in range(len(RMS_Y)):
    if RMS_Y[i]==1:
        F_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        F_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        F_RMS_bio.append(RMS_bio[i:i+1,:])
        F_RMS_Y.append(RMS_Y[i:i+1])
    if RMS_Y[i]==0:
        nonF_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        nonF_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        nonF_RMS_bio.append(RMS_bio[i:i+1,:])
        nonF_RMS_Y.append(RMS_Y[i:i+1]) 
F_RMS_accel_statistics = np.concatenate(F_RMS_accel_statistics, axis=0)
F_RMS_gyro_statistics = np.concatenate(F_RMS_gyro_statistics, axis=0)
F_RMS_bio = np.concatenate(F_RMS_bio, axis=0)
F_RMS_Y = np.concatenate(F_RMS_Y, axis=0)
nonF_RMS_accel_statistics = np.concatenate(nonF_RMS_accel_statistics, axis=0)
nonF_RMS_gyro_statistics = np.concatenate(nonF_RMS_gyro_statistics, axis=0)
nonF_RMS_bio = np.concatenate(nonF_RMS_bio, axis=0)
nonF_RMS_Y = np.concatenate(nonF_RMS_Y, axis=0)

    ##################### Fallers ############################
        #-------  Accelerometer Features -------
F_AccX_mean_B, F_AccX_std_B, F_AccX_CV_B, F_AccX_min_B, F_AccX_max_B = F_accel_statistics[:,0], F_accel_statistics[:,6], F_accel_statistics[:,3], F_accel_statistics[:,9], F_accel_statistics[:,12]
F_AccY_mean_B, F_AccY_std_B, F_AccY_CV_B, F_AccY_min_B, F_AccY_max_B = F_accel_statistics[:,1], F_accel_statistics[:,7], F_accel_statistics[:,4], F_accel_statistics[:,10], F_accel_statistics[:,13]
F_AccZ_mean_B, F_AccZ_std_B, F_AccZ_CV_B, F_AccZ_min_B, F_AccZ_max_B = F_accel_statistics[:,2], F_accel_statistics[:,8], F_accel_statistics[:,5], F_accel_statistics[:,11], F_accel_statistics[:,14]
F_RMS_Acc_mean_B, F_RMS_Acc_std_B, F_RMS_Acc_CV_B, F_RMS_Acc_min_B, F_RMS_Acc_max_B = F_RMS_accel_statistics[:,0], F_RMS_accel_statistics[:,1], F_RMS_accel_statistics[:,2], F_RMS_accel_statistics[:,3], F_RMS_accel_statistics[:,4]
       #-------  Gyroscope Features -------
F_GyrX_mean_B, F_GyrX_std_B, F_GyrX_CV_B, F_GyrX_min_B, F_GyrX_max_B = F_gyro_statistics[:,0], F_gyro_statistics[:,6], F_gyro_statistics[:,3], F_gyro_statistics[:,9], F_gyro_statistics[:,12]
F_GyrY_mean_B, F_GyrY_std_B, F_GyrY_CV_B, F_GyrY_min_B, F_GyrY_max_B = F_gyro_statistics[:,1], F_gyro_statistics[:,7], F_gyro_statistics[:,4], F_gyro_statistics[:,10], F_gyro_statistics[:,13]
F_GyrZ_mean_B, F_GyrZ_std_B, F_GyrZ_CV_B, F_GyrZ_min_B, F_GyrZ_max_B = F_gyro_statistics[:,2], F_gyro_statistics[:,8], F_gyro_statistics[:,5], F_gyro_statistics[:,11], F_gyro_statistics[:,14]
F_RMS_Gyr_mean_B, F_RMS_Gyr_std_B, F_RMS_Gyr_CV_B, F_RMS_Gyr_min_B, F_RMS_Gyr_max_B = F_RMS_gyro_statistics[:,0], F_RMS_gyro_statistics[:,1], F_RMS_gyro_statistics[:,2], F_RMS_gyro_statistics[:,3], F_RMS_gyro_statistics[:,4]

    ##################### Non-fallers ############################
        #-------  Accelerometer Features -------
nonF_AccX_mean_B, nonF_AccX_std_B, nonF_AccX_CV_B, nonF_AccX_min_B, nonF_AccX_max_B = nonF_accel_statistics[:,0], nonF_accel_statistics[:,6], nonF_accel_statistics[:,3], nonF_accel_statistics[:,9], nonF_accel_statistics[:,12]
nonF_AccY_mean_B, nonF_AccY_std_B, nonF_AccY_CV_B, nonF_AccY_min_B, nonF_AccY_max_B = nonF_accel_statistics[:,1], nonF_accel_statistics[:,7], nonF_accel_statistics[:,4], nonF_accel_statistics[:,10], nonF_accel_statistics[:,13]
nonF_AccZ_mean_B, nonF_AccZ_std_B, nonF_AccZ_CV_B, nonF_AccZ_min_B, nonF_AccZ_max_B = nonF_accel_statistics[:,2], nonF_accel_statistics[:,8], nonF_accel_statistics[:,5], nonF_accel_statistics[:,11], nonF_accel_statistics[:,14]
nonF_RMS_Acc_mean_B, nonF_RMS_Acc_std_B, nonF_RMS_Acc_CV_B, nonF_RMS_Acc_min_B, nonF_RMS_Acc_max_B = nonF_RMS_accel_statistics[:,0], nonF_RMS_accel_statistics[:,1], nonF_RMS_accel_statistics[:,2], nonF_RMS_accel_statistics[:,3], nonF_RMS_accel_statistics[:,4]
         #-------  Gyroscope Features -------
nonF_GyrX_mean_B, nonF_GyrX_std_B, nonF_GyrX_CV_B, nonF_GyrX_min_B, nonF_GyrX_max_B = nonF_gyro_statistics[:,0], nonF_gyro_statistics[:,6], nonF_gyro_statistics[:,3], nonF_gyro_statistics[:,9], nonF_gyro_statistics[:,12]
nonF_GyrY_mean_B, nonF_GyrY_std_B, nonF_GyrY_CV_B, nonF_GyrY_min_B, nonF_GyrY_max_B = nonF_gyro_statistics[:,1], nonF_gyro_statistics[:,7], nonF_gyro_statistics[:,4], nonF_gyro_statistics[:,10], nonF_gyro_statistics[:,13]
nonF_GyrZ_mean_B, nonF_GyrZ_std_B, nonF_GyrZ_CV_B, nonF_GyrZ_min_B, nonF_GyrZ_max_B = nonF_gyro_statistics[:,2], nonF_gyro_statistics[:,8], nonF_gyro_statistics[:,5], nonF_gyro_statistics[:,11], nonF_gyro_statistics[:,14]
nonF_RMS_Gyr_mean_B, nonF_RMS_Gyr_std_B, nonF_RMS_Gyr_CV_B, nonF_RMS_Gyr_min_B, nonF_RMS_Gyr_max_B = nonF_RMS_gyro_statistics[:,0], nonF_RMS_gyro_statistics[:,1], nonF_RMS_gyro_statistics[:,2], nonF_RMS_gyro_statistics[:,3], nonF_RMS_gyro_statistics[:,4]

########################################################################################################################################################################
sensor = 'Right'
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\' + sensor + '\\'

accel_statistics = np.load(filepath+sensor+'accel_statistics.npy')
gyro_statistics = np.load(filepath+sensor+'gyro_statistics.npy')
bio = np.load(filepath+sensor+'bio.npy')
Y = np.load(filepath+sensor+'Y.npy')
subject_number = np.load(filepath+sensor+'subject_number.npy')
F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_bio, nonF_bio = [], [], [], [], [], []
F_Y, nonF_Y, F_subject_number, nonF_subject_number = [], [], [], []
for i in range(len(Y)):
    if Y[i]==1:
        F_accel_statistics.append(accel_statistics[i:i+1,:])
        F_gyro_statistics.append(gyro_statistics[i:i+1,:])
        F_bio.append(bio[i:i+1,:])
        F_Y.append(Y[i:i+1])
        F_subject_number.append(subject_number[i:i+1,:])
    if Y[i]==0:
        nonF_accel_statistics.append(accel_statistics[i:i+1,:])
        nonF_gyro_statistics.append(gyro_statistics[i:i+1,:])
        nonF_bio.append(bio[i:i+1,:])
        nonF_Y.append(Y[i:i+1])
        nonF_subject_number.append(subject_number[i:i+1,:])        
F_accel_statistics = np.concatenate(F_accel_statistics, axis=0)
F_gyro_statistics = np.concatenate(F_gyro_statistics, axis=0)
F_bio = np.concatenate(F_bio, axis=0)
F_Y = np.concatenate(F_Y, axis=0)
F_subject_number = np.concatenate(F_subject_number, axis=0)
nonF_accel_statistics = np.concatenate(nonF_accel_statistics, axis=0)
nonF_gyro_statistics = np.concatenate(nonF_gyro_statistics, axis=0)
nonF_bio = np.concatenate(nonF_bio, axis=0)
nonF_Y = np.concatenate(nonF_Y, axis=0)
nonF_subject_number = np.concatenate(nonF_subject_number, axis=0)

            ##################
RMS_bio = np.load(filepath+sensor+'RMS_bio.npy')
RMS_Y = np.load(filepath+sensor+'RMS_Y.npy')
RMS_accel_statistics = np.load(filepath+sensor+'RMS_accel_statistics.npy')
RMS_gyro_statistics= np.load(filepath+sensor+'RMS_gyro_statistics.npy')

F_RMS_accel_statistics, nonF_RMS_accel_statistics, F_RMS_gyro_statistics, nonF_RMS_gyro_statistics, F_RMS_bio, nonF_RMS_bio = [], [], [], [], [], []
F_RMS_Y, nonF_RMS_Y = [], []
for i in range(len(RMS_Y)):
    if RMS_Y[i]==1:
        F_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        F_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        F_RMS_bio.append(RMS_bio[i:i+1,:])
        F_RMS_Y.append(RMS_Y[i:i+1])
    if RMS_Y[i]==0:
        nonF_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        nonF_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        nonF_RMS_bio.append(RMS_bio[i:i+1,:])
        nonF_RMS_Y.append(RMS_Y[i:i+1]) 
F_RMS_accel_statistics = np.concatenate(F_RMS_accel_statistics, axis=0)
F_RMS_gyro_statistics = np.concatenate(F_RMS_gyro_statistics, axis=0)
F_RMS_bio = np.concatenate(F_RMS_bio, axis=0)
F_RMS_Y = np.concatenate(F_RMS_Y, axis=0)
nonF_RMS_accel_statistics = np.concatenate(nonF_RMS_accel_statistics, axis=0)
nonF_RMS_gyro_statistics = np.concatenate(nonF_RMS_gyro_statistics, axis=0)
nonF_RMS_bio = np.concatenate(nonF_RMS_bio, axis=0)
nonF_RMS_Y = np.concatenate(nonF_RMS_Y, axis=0)


    ##################### Fallers ############################
        #-------  Accelerometer Features -------
F_AccX_mean_R, F_AccX_std_R, F_AccX_CV_R, F_AccX_min_R, F_AccX_max_R = F_accel_statistics[:,0], F_accel_statistics[:,6], F_accel_statistics[:,3], F_accel_statistics[:,9], F_accel_statistics[:,12]
F_AccY_mean_R, F_AccY_std_R, F_AccY_CV_R, F_AccY_min_R, F_AccY_max_R = F_accel_statistics[:,1], F_accel_statistics[:,7], F_accel_statistics[:,4], F_accel_statistics[:,10], F_accel_statistics[:,13]
F_AccZ_mean_R, F_AccZ_std_R, F_AccZ_CV_R, F_AccZ_min_R, F_AccZ_max_R = F_accel_statistics[:,2], F_accel_statistics[:,8], F_accel_statistics[:,5], F_accel_statistics[:,11], F_accel_statistics[:,14]
F_RMS_Acc_mean_R, F_RMS_Acc_std_R, F_RMS_Acc_CV_R, F_RMS_Acc_min_R, F_RMS_Acc_max_R = F_RMS_accel_statistics[:,0], F_RMS_accel_statistics[:,1], F_RMS_accel_statistics[:,2], F_RMS_accel_statistics[:,3], F_RMS_accel_statistics[:,4]
         #-------  Gyroscope Features -------
F_GyrX_mean_R, F_GyrX_std_R, F_GyrX_CV_R, F_GyrX_min_R, F_GyrX_max_R = F_gyro_statistics[:,0], F_gyro_statistics[:,6], F_gyro_statistics[:,3], F_gyro_statistics[:,9], F_gyro_statistics[:,12]
F_GyrY_mean_R, F_GyrY_std_R, F_GyrY_CV_R, F_GyrY_min_R, F_GyrY_max_R = F_gyro_statistics[:,1], F_gyro_statistics[:,7], F_gyro_statistics[:,4], F_gyro_statistics[:,10], F_gyro_statistics[:,13]
F_GyrZ_mean_R, F_GyrZ_std_R, F_GyrZ_CV_R, F_GyrZ_min_R, F_GyrZ_max_R = F_gyro_statistics[:,2], F_gyro_statistics[:,8], F_gyro_statistics[:,5], F_gyro_statistics[:,11], F_gyro_statistics[:,14]
F_RMS_Gyr_mean_R, F_RMS_Gyr_std_R, F_RMS_Gyr_CV_R, F_RMS_Gyr_min_R, F_RMS_Gyr_max_R = F_RMS_gyro_statistics[:,0], F_RMS_gyro_statistics[:,1], F_RMS_gyro_statistics[:,2], F_RMS_gyro_statistics[:,3], F_RMS_gyro_statistics[:,4]

    ##################### Non-fallers ############################
        #-------  Accelerometer Features -------
nonF_AccX_mean_R, nonF_AccX_std_R, nonF_AccX_CV_R, nonF_AccX_min_R, nonF_AccX_max_R = nonF_accel_statistics[:,0], nonF_accel_statistics[:,6], nonF_accel_statistics[:,3], nonF_accel_statistics[:,9], nonF_accel_statistics[:,12]
nonF_AccY_mean_R, nonF_AccY_std_R, nonF_AccY_CV_R, nonF_AccY_min_R, nonF_AccY_max_R = nonF_accel_statistics[:,1], nonF_accel_statistics[:,7], nonF_accel_statistics[:,4], nonF_accel_statistics[:,10], nonF_accel_statistics[:,13]
nonF_AccZ_mean_R, nonF_AccZ_std_R, nonF_AccZ_CV_R, nonF_AccZ_min_R, nonF_AccZ_max_R = nonF_accel_statistics[:,2], nonF_accel_statistics[:,8], nonF_accel_statistics[:,5], nonF_accel_statistics[:,11], nonF_accel_statistics[:,14]
nonF_RMS_Acc_mean_R, nonF_RMS_Acc_std_R, nonF_RMS_Acc_CV_R, nonF_RMS_Acc_min_R, nonF_RMS_Acc_max_R = nonF_RMS_accel_statistics[:,0], nonF_RMS_accel_statistics[:,1], nonF_RMS_accel_statistics[:,2], nonF_RMS_accel_statistics[:,3], nonF_RMS_accel_statistics[:,4]
         #-------  Gyroscope Features -------
nonF_GyrX_mean_R, nonF_GyrX_std_R, nonF_GyrX_CV_R, nonF_GyrX_min_R, nonF_GyrX_max_R = nonF_gyro_statistics[:,0], nonF_gyro_statistics[:,6], nonF_gyro_statistics[:,3], nonF_gyro_statistics[:,9], nonF_gyro_statistics[:,12]
nonF_GyrY_mean_R, nonF_GyrY_std_R, nonF_GyrY_CV_R, nonF_GyrY_min_R, nonF_GyrY_max_R = nonF_gyro_statistics[:,1], nonF_gyro_statistics[:,7], nonF_gyro_statistics[:,4], nonF_gyro_statistics[:,10], nonF_gyro_statistics[:,13]
nonF_GyrZ_mean_R, nonF_GyrZ_std_R, nonF_GyrZ_CV_R, nonF_GyrZ_min_R, nonF_GyrZ_max_R = nonF_gyro_statistics[:,2], nonF_gyro_statistics[:,8], nonF_gyro_statistics[:,5], nonF_gyro_statistics[:,11], nonF_gyro_statistics[:,14]
nonF_RMS_Gyr_mean_R, nonF_RMS_Gyr_std_R, nonF_RMS_Gyr_CV_R, nonF_RMS_Gyr_min_R, nonF_RMS_Gyr_max_R = nonF_RMS_gyro_statistics[:,0], nonF_RMS_gyro_statistics[:,1], nonF_RMS_gyro_statistics[:,2], nonF_RMS_gyro_statistics[:,3], nonF_RMS_gyro_statistics[:,4]


sensor = 'Left'
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\' + sensor + '\\'

accel_statistics = np.load(filepath+sensor+'accel_statistics.npy')
gyro_statistics = np.load(filepath+sensor+'gyro_statistics.npy')
bio = np.load(filepath+sensor+'bio.npy')
Y = np.load(filepath+sensor+'Y.npy')
subject_number = np.load(filepath+sensor+'subject_number.npy')
F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_bio, nonF_bio = [], [], [], [], [], []
F_Y, nonF_Y, F_subject_number, nonF_subject_number = [], [], [], []
for i in range(len(Y)):
    if Y[i]==1:
        F_accel_statistics.append(accel_statistics[i:i+1,:])
        F_gyro_statistics.append(gyro_statistics[i:i+1,:])
        F_bio.append(bio[i:i+1,:])
        F_Y.append(Y[i:i+1])
        F_subject_number.append(subject_number[i:i+1,:])
    if Y[i]==0:
        nonF_accel_statistics.append(accel_statistics[i:i+1,:])
        nonF_gyro_statistics.append(gyro_statistics[i:i+1,:])
        nonF_bio.append(bio[i:i+1,:])
        nonF_Y.append(Y[i:i+1])
        nonF_subject_number.append(subject_number[i:i+1,:])        
F_accel_statistics = np.concatenate(F_accel_statistics, axis=0)
F_gyro_statistics = np.concatenate(F_gyro_statistics, axis=0)
F_bio = np.concatenate(F_bio, axis=0)
F_Y = np.concatenate(F_Y, axis=0)
F_subject_number = np.concatenate(F_subject_number, axis=0)
nonF_accel_statistics = np.concatenate(nonF_accel_statistics, axis=0)
nonF_gyro_statistics = np.concatenate(nonF_gyro_statistics, axis=0)
nonF_bio = np.concatenate(nonF_bio, axis=0)
nonF_Y = np.concatenate(nonF_Y, axis=0)
nonF_subject_number = np.concatenate(nonF_subject_number, axis=0)

            ##################
RMS_bio = np.load(filepath+sensor+'RMS_bio.npy')
RMS_Y = np.load(filepath+sensor+'RMS_Y.npy')
RMS_accel_statistics = np.load(filepath+sensor+'RMS_accel_statistics.npy')
RMS_gyro_statistics= np.load(filepath+sensor+'RMS_gyro_statistics.npy')

F_RMS_accel_statistics, nonF_RMS_accel_statistics, F_RMS_gyro_statistics, nonF_RMS_gyro_statistics, F_RMS_bio, nonF_RMS_bio = [], [], [], [], [], []
F_RMS_Y, nonF_RMS_Y = [], []
for i in range(len(RMS_Y)):
    if RMS_Y[i]==1:
        F_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        F_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        F_RMS_bio.append(RMS_bio[i:i+1,:])
        F_RMS_Y.append(RMS_Y[i:i+1])
    if RMS_Y[i]==0:
        nonF_RMS_accel_statistics.append(RMS_accel_statistics[i:i+1,:])
        nonF_RMS_gyro_statistics.append(RMS_gyro_statistics[i:i+1,:])
        nonF_RMS_bio.append(RMS_bio[i:i+1,:])
        nonF_RMS_Y.append(RMS_Y[i:i+1]) 
F_RMS_accel_statistics = np.concatenate(F_RMS_accel_statistics, axis=0)
F_RMS_gyro_statistics = np.concatenate(F_RMS_gyro_statistics, axis=0)
F_RMS_bio = np.concatenate(F_RMS_bio, axis=0)
F_RMS_Y = np.concatenate(F_RMS_Y, axis=0)
nonF_RMS_accel_statistics = np.concatenate(nonF_RMS_accel_statistics, axis=0)
nonF_RMS_gyro_statistics = np.concatenate(nonF_RMS_gyro_statistics, axis=0)
nonF_RMS_bio = np.concatenate(nonF_RMS_bio, axis=0)
nonF_RMS_Y = np.concatenate(nonF_RMS_Y, axis=0)

    ##################### Fallers ############################
        #-------  Accelerometer Features -------
F_AccX_mean_L, F_AccX_std_L, F_AccX_CV_L, F_AccX_min_L, F_AccX_max_L = F_accel_statistics[:,0], F_accel_statistics[:,6], F_accel_statistics[:,3], F_accel_statistics[:,9], F_accel_statistics[:,12]
F_AccY_mean_L, F_AccY_std_L, F_AccY_CV_L, F_AccY_min_L, F_AccY_max_L = F_accel_statistics[:,1], F_accel_statistics[:,7], F_accel_statistics[:,4], F_accel_statistics[:,10], F_accel_statistics[:,13]
F_AccZ_mean_L, F_AccZ_std_L, F_AccZ_CV_L, F_AccZ_min_L, F_AccZ_max_L = F_accel_statistics[:,2], F_accel_statistics[:,8], F_accel_statistics[:,5], F_accel_statistics[:,11], F_accel_statistics[:,14]
F_RMS_Acc_mean_L, F_RMS_Acc_std_L, F_RMS_Acc_CV_L, F_RMS_Acc_min_L, F_RMS_Acc_max_L = F_RMS_accel_statistics[:,0], F_RMS_accel_statistics[:,1], F_RMS_accel_statistics[:,2], F_RMS_accel_statistics[:,3], F_RMS_accel_statistics[:,4]
         #-------  Gyroscope Features -------
F_GyrX_mean_L, F_GyrX_std_L, F_GyrX_CV_L, F_GyrX_min_L, F_GyrX_max_L = F_gyro_statistics[:,0], F_gyro_statistics[:,6], F_gyro_statistics[:,3], F_gyro_statistics[:,9], F_gyro_statistics[:,12]
F_GyrY_mean_L, F_GyrY_std_L, F_GyrY_CV_L, F_GyrY_min_L, F_GyrY_max_L = F_gyro_statistics[:,1], F_gyro_statistics[:,7], F_gyro_statistics[:,4], F_gyro_statistics[:,10], F_gyro_statistics[:,13]
F_GyrZ_mean_L, F_GyrZ_std_L, F_GyrZ_CV_L, F_GyrZ_min_L, F_GyrZ_max_L = F_gyro_statistics[:,2], F_gyro_statistics[:,8], F_gyro_statistics[:,5], F_gyro_statistics[:,11], F_gyro_statistics[:,14]
F_RMS_Gyr_mean_L, F_RMS_Gyr_std_L, F_RMS_Gyr_CV_L, F_RMS_Gyr_min_L, F_RMS_Gyr_max_L = F_RMS_gyro_statistics[:,0], F_RMS_gyro_statistics[:,1], F_RMS_gyro_statistics[:,2], F_RMS_gyro_statistics[:,3], F_RMS_gyro_statistics[:,4]

    ##################### Non-fallers ############################
        #-------  Accelerometer Features -------
nonF_AccX_mean_L, nonF_AccX_std_L, nonF_AccX_CV_L, nonF_AccX_min_L, nonF_AccX_max_L = nonF_accel_statistics[:,0], nonF_accel_statistics[:,6], nonF_accel_statistics[:,3], nonF_accel_statistics[:,9], nonF_accel_statistics[:,12]
nonF_AccY_mean_L, nonF_AccY_std_L, nonF_AccY_CV_L, nonF_AccY_min_L, nonF_AccY_max_L = nonF_accel_statistics[:,1], nonF_accel_statistics[:,7], nonF_accel_statistics[:,4], nonF_accel_statistics[:,10], nonF_accel_statistics[:,13]
nonF_AccZ_mean_L, nonF_AccZ_std_L, nonF_AccZ_CV_L, nonF_AccZ_min_L, nonF_AccZ_max_L = nonF_accel_statistics[:,2], nonF_accel_statistics[:,8], nonF_accel_statistics[:,5], nonF_accel_statistics[:,11], nonF_accel_statistics[:,14]
nonF_RMS_Acc_mean_L, nonF_RMS_Acc_std_L, nonF_RMS_Acc_CV_L, nonF_RMS_Acc_min_L, nonF_RMS_Acc_max_L = nonF_RMS_accel_statistics[:,0], nonF_RMS_accel_statistics[:,1], nonF_RMS_accel_statistics[:,2], nonF_RMS_accel_statistics[:,3], nonF_RMS_accel_statistics[:,4]
        #-------  Gyroscope Features -------
nonF_GyrX_mean_L, nonF_GyrX_std_L, nonF_GyrX_CV_L, nonF_GyrX_min_L, nonF_GyrX_max_L = nonF_gyro_statistics[:,0], nonF_gyro_statistics[:,6], nonF_gyro_statistics[:,3], nonF_gyro_statistics[:,9], nonF_gyro_statistics[:,12]
nonF_GyrY_mean_L, nonF_GyrY_std_L, nonF_GyrY_CV_L, nonF_GyrY_min_L, nonF_GyrY_max_L = nonF_gyro_statistics[:,1], nonF_gyro_statistics[:,7], nonF_gyro_statistics[:,4], nonF_gyro_statistics[:,10], nonF_gyro_statistics[:,13]
nonF_GyrZ_mean_L, nonF_GyrZ_std_L, nonF_GyrZ_CV_L, nonF_GyrZ_min_L, nonF_GyrZ_max_L = nonF_gyro_statistics[:,2], nonF_gyro_statistics[:,8], nonF_gyro_statistics[:,5], nonF_gyro_statistics[:,11], nonF_gyro_statistics[:,14]
nonF_RMS_Gyr_mean_L, nonF_RMS_Gyr_std_L, nonF_RMS_Gyr_CV_L, nonF_RMS_Gyr_min_L, nonF_RMS_Gyr_max_L = nonF_RMS_gyro_statistics[:,0], nonF_RMS_gyro_statistics[:,1], nonF_RMS_gyro_statistics[:,2], nonF_RMS_gyro_statistics[:,3], nonF_RMS_gyro_statistics[:,4]


##### Load some other attributes
import pandas as pd
codebook = pd.read_excel("C:\\Users\\Venus\\Desktop\\Comps\\Codebook and clinical evaluations\\walk data 7 7 2020 with original questions Venous.xlsx")     
codebook = codebook[['Fall', 'RiskAssessmentScore', 'Gender', 'BMI','NumDxs', 'NumRxs', 'NumPsychoactiveRxs', 'TUGTime', 'ChairStand', 'TotalBalanceScore']][2:102]
Y = codebook[['Fall']].to_numpy()
BMI = codebook[['BMI']].to_numpy()
Dxs = codebook[['NumDxs']].to_numpy()
Rxs = codebook[['NumRxs']].to_numpy()
PsychoRxs = codebook[['NumPsychoactiveRxs']].to_numpy()
RAS = codebook[['RiskAssessmentScore']].to_numpy()
TUG = codebook[['TUGTime']].to_numpy()
Balance = codebook[['TotalBalanceScore']].to_numpy()
Stands = codebook[['ChairStand']].to_numpy()
Gender = codebook[['Gender']].to_numpy()
# print(TUG)
# print(BMI.shape)

F_Gender, F_BMI, F_Dxs, F_Rxs, F_PsychoRxs = [], [], [], [], []
nonF_Gender, nonF_BMI, nonF_Dxs, nonF_Rxs, nonF_PsychoRxs = [], [], [], [], []
F_RAS, F_TUG, F_Balance, F_Stands = [], [], [], []
nonF_RAS, nonF_TUG, nonF_Balance, nonF_Stands = [], [], [], []

for i in range(len(Y)):
    if Y[i] == 1:
        F_Gender.append(Gender[i,0])  
        F_BMI.append(BMI[i,0])   
        F_Dxs.append(Dxs[i,0])    
        F_Rxs.append(Rxs[i,0])      
        F_PsychoRxs.append(PsychoRxs[i,0])       
        F_RAS.append(RAS[i,0])
        F_TUG.append(TUG[i,0])
        F_Balance.append(Balance[i,0]) 
        F_Stands.append(Stands[i,0])
    if Y[i] == 0:
        nonF_Gender.append(Gender[i,0])
        nonF_BMI.append(BMI[i,0])
        nonF_Dxs.append(Dxs[i,0])   
        nonF_Rxs.append(Rxs[i,0])
        nonF_PsychoRxs.append(PsychoRxs[i,0])       
        nonF_RAS.append(RAS[i,0])
        nonF_TUG.append(TUG[i,0])
        nonF_Balance.append(Balance[i,0])
        nonF_Stands.append(Stands[i,0])
        
F_Gender = np.asarray(F_Gender)
F_BMI = np.asarray(F_BMI)
F_Dxs = np.asarray(F_Dxs)
F_Rxs = np.asarray(F_Rxs)   
F_PsychoRxs = np.asarray(F_PsychoRxs)
F_RAS = np.asarray(F_RAS)
F_TUG = np.asarray(F_TUG)
F_Balance = np.asarray(F_Balance)
F_Stands = np.asarray(F_Stands)

nonF_Gender = np.asarray(nonF_Gender)
nonF_BMI = np.asarray(nonF_BMI) 
nonF_Dxs = np.asarray(nonF_Dxs)
nonF_Rxs = np.asarray(nonF_Rxs)
nonF_PsychoRxs = np.asarray(nonF_PsychoRxs)
nonF_RAS = np.asarray(nonF_RAS)
nonF_TUG = np.asarray(nonF_TUG)
nonF_Balance = np.asarray(nonF_Balance)
nonF_Stands = np.asarray(nonF_Stands)
print(nonF_BMI.shape)
print(np.sum(Y))
       
bio_Features = {'Age': [F_age, nonF_age] ,'Gender': [F_Gender, nonF_Gender], 'BMI': [F_BMI, nonF_BMI],
                'Dxs': [F_Dxs, nonF_Dxs], 'Rxs': [F_Rxs, nonF_Rxs], 'PsychoRxs': [F_PsychoRxs, nonF_PsychoRxs],
                'RAS': [F_RAS, nonF_RAS], 'TUG': [F_TUG, nonF_TUG], 'Balance': [F_Balance, nonF_Balance], 'Stands': [F_Stands, nonF_Stands]}

Kinematic_Features =  {'AccX_CV': [F_AccX_CV_B, nonF_AccX_CV_B, F_AccX_CV_R, nonF_AccX_CV_R, F_AccX_CV_L, nonF_AccX_CV_L],
                    'AccY_CV': [F_AccY_CV_B, nonF_AccY_CV_B, F_AccY_CV_R, nonF_AccY_CV_R, F_AccY_CV_L, nonF_AccY_CV_L],
                    'AccZ_CV': [F_AccZ_CV_B, nonF_AccZ_CV_B, F_AccZ_CV_R, nonF_AccZ_CV_R, F_AccZ_CV_L, nonF_AccZ_CV_L],
                    'GyrX_CV': [F_GyrX_CV_B, nonF_GyrX_CV_B, F_GyrX_CV_R, nonF_GyrX_CV_R, F_GyrX_CV_L, nonF_GyrX_CV_L],
                    'GyrY_CV': [F_GyrY_CV_B, nonF_GyrY_CV_B, F_GyrY_CV_R, nonF_GyrY_CV_R, F_GyrY_CV_L, nonF_GyrY_CV_L],
                    'GyrZ_CV': [F_GyrZ_CV_B, nonF_GyrZ_CV_B, F_GyrZ_CV_R, nonF_GyrZ_CV_R, F_GyrZ_CV_L, nonF_GyrZ_CV_L],
                    'Acc_RMS_mean': [F_RMS_Acc_mean_B, nonF_RMS_Acc_mean_B, F_RMS_Acc_mean_R, nonF_RMS_Acc_mean_R, F_RMS_Acc_mean_L, nonF_RMS_Acc_mean_L],
                    'Acc_RMS_std': [F_RMS_Acc_std_B, nonF_RMS_Acc_std_B, F_RMS_Acc_std_R, nonF_RMS_Acc_std_R, F_RMS_Acc_std_L, nonF_RMS_Acc_std_L],
                    'Acc_RMS_CV': [F_RMS_Acc_CV_B, nonF_RMS_Acc_CV_B, F_RMS_Acc_CV_R, nonF_RMS_Acc_CV_R, F_RMS_Acc_CV_L, nonF_RMS_Acc_CV_L],
                    'Gyr_RMS_mean': [F_RMS_Gyr_mean_B, nonF_RMS_Gyr_mean_B, F_RMS_Gyr_mean_R, nonF_RMS_Gyr_mean_R, F_RMS_Gyr_mean_L, nonF_RMS_Gyr_mean_L],
                    'Gyr_RMS_std': [F_RMS_Gyr_std_B, nonF_RMS_Gyr_std_B, F_RMS_Gyr_std_R, nonF_RMS_Gyr_std_R, F_RMS_Gyr_std_L, nonF_RMS_Gyr_std_L],
                    'Gyr_RMS_CV': [F_RMS_Gyr_CV_B, nonF_RMS_Gyr_CV_B, F_RMS_Gyr_CV_R, nonF_RMS_Gyr_CV_R, F_RMS_Gyr_CV_L, nonF_RMS_Gyr_CV_L]}
###-------------------------------------------------------------------------------------------------------------------------------------------------------###
###------------------------------------------------ STATMODELS Logit ------------------------------------------------ 
###-------------------------------------------------------------------------------------------------------------------------------------------------------###

####################--------- z-test for all the features using statsmodels  ----------########################
""" Note: statsmodels.stats.weightstats.ztest() can only handle usevar='pooled'
    - My samples are mean, std, CV, min or max of acceleration and gyroscope time series for each subject.
    - Two-sample case: H1: Mean of F_AccX_mean_L < OR > Mean of nonF_AccX_mean_L,
    - Value is the difference between mean of x1 and mean of x2 under the Null hypothesis.
    - alternative
        The alternative hypothesis, H1, has to be one of the following ‘two-sided’: H1: difference in means not equal to value (default) ‘larger’ : H1: difference in means larger than value ‘smaller’ : H1: difference in means smaller than value
    - usevar, ‘pooled’ or ‘unequal’
        If pooled, then the standard deviation of the samples is assumed to be the same. If unequal, then the standard deviations of the samples may be different.
"""

def perform_ztest(Features, usevar, value):
    
    if Features == Kinematic_Features:
        for sensor in ['Back', 'Right', 'Left']:
            if sensor == 'Back':
                print("\t\t\t\t\t \t\t\t\t\t\t" + "--------------------------------------- Back Sensor ---------------------------------------")
                sensor_name = '_B'
                [idx_F, idx_nonF] = [0, 1]
            elif sensor == 'Right':
                print("\t\t\t\t\t \t\t\t\t\t\t" + "--------------------------------------- Right Sensor ---------------------------------------")
                sensor_name = '_R'
                [idx_F, idx_nonF] = [2, 3]
            else:
                print("\t\t\t\t\t \t\t\t\t\t\t" + "--------------------------------------- Left Sensor ---------------------------------------")
                sensor_name = '_L'
                [idx_F, idx_nonF] = [4, 5]
                        
            for feature in Features:
                print("\t\t\t\t\t \t\t\t\t\t\t" + '----------' + feature + sensor_name + '----------')
                for alternative in ['two-sided', 'smaller', 'larger']:
                    S1 = weightstats.DescrStatsW(Features[feature][idx_F])
                    S2 = weightstats.DescrStatsW(Features[feature][idx_nonF])
                    CM = weightstats.CompareMeans(S1, S2)    
                    zstat, p_value = CM.ztest_ind(alternative=alternative, usevar=usevar, value=value)
                    
                    lower, upper = CM.zconfint_diff(alpha=0.05, alternative=alternative, usevar=usevar)
                    if alternative == 'two-sided':
                        H0 = 'equal to'
                    elif alternative == 'smaller':
                        H0 ='larger than or equal to'
                    else:
                        H0 = 'smaller than or equal to'                 
                    if p_value < 0.05:
                        print('P_value is %.5f. We reject the null hypothesis, so, the mean of fallers ' %p_value + feature + sensor_name +' is not '  + H0 + ' mean of nonfallers ' + feature )
                        print('%%95 CI (%.3f, %.3f)' %(lower, upper))
                        
                    else:
                        print('P_value is %.5f. We cannot reject the null hypothesis and the mean of fallers ' %p_value + feature + sensor_name +' is '  + H0 + ' mean of nonfallers ' + feature)
                        print('%%95 CI (%.3f, %.3f)' %(lower, upper))
                        
                print('---------------------------')
            print('\n')
            print('---------------------------------------------------------------------------------------------------------------------------------------------')
    else:
        for feature in Features:
            print("\t\t\t\t\t \t\t\t\t\t\t" + '----------' + feature + '----------')
            for alternative in ['two-sided', 'smaller', 'larger']:
                S1 = weightstats.DescrStatsW(Features[feature][0])
                S2 = weightstats.DescrStatsW(Features[feature][1])
                CM = weightstats.CompareMeans(S1, S2)    
                zstat, p_value = CM.ztest_ind(alternative=alternative, usevar=usevar, value=value)
                
                lower, upper = CM.zconfint_diff(alpha=0.05, alternative=alternative, usevar=usevar)
                
                
                if alternative == 'two-sided':
                    H0 = 'equal to'
                elif alternative == 'smaller':
                    H0 ='larger than or equal to'
                else:
                    H0 = 'smaller than or equal to'
             
                if p_value < 0.05:
                    print('P_value is %.5f. We reject the null hypothesis, so, the mean of fallers ' %p_value + feature +' is not '  + H0 + ' mean of nonfallers ' + feature )
                    print('%%95 CI (%.3f, %.3f)' %(lower, upper))
                    print('----')
                else:
                    print('P_value is %.5f. We cannot reject the null hypothesis and the mean of fallers ' %p_value + feature +' is '  + H0 + ' mean of nonfallers ' + feature)
                    print('%%95 CI (%.3f, %.3f)' %(lower, upper))
                    print('----')
            print('\n')
        print('---------------------------------------------------------------------------------------')
        

#### z-test for kinematics features
perform_ztest(Kinematic_Features, usevar='unequal', value=0)

### Compare p-values and find the group sensitive features, then perform ROC analysis (C-statistics on the group sensitive features),
### This means comparing AUCs and find which feature has an overall stronger ability to discriminae between groups.

################# t-test using scipy ########################

# # from scipy.stats import ttest_ind # from scipy.stats import ttest_ind_from_stats
# ##### T-test for means of two independent samples from descriptive statistics. A two-sided test.
# # t_statistics, p_value = ttest_ind_from_stats(np.mean(F_AccX_mean_B), np.std(F_AccX_mean_B), len(F_patient_num),
# #                                              np.mean(nonF_AccX_mean_B), np.std(nonF_AccX_mean_B), len(nonF_patient_num), equal_var=False)
# # p_value = ttest_ind_from_stats(np.mean(F_AccX_mean_B), np.std(F_AccX_mean_B), len(F_patient_num),
# #                                np.mean(nonF_AccX_mean_B), np.std(nonF_AccX_mean_B), len(nonF_patient_num), equal_var=False).pvalue
# # t_statistics, p_value = ttest_ind(F_AccX_mean_B, nonF_AccX_mean_B, equal_var=False) #### This test assumes that the populations have identical variances by default.
####





 
