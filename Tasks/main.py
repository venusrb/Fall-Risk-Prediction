# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:56:37 2019

@author: vroshdibenam
"""

# Set path to the project root folder, in order to read `data` module.
import os, sys
import data
import numpy as np
########################################################################
def Print_statistics(data):
    if len(sensor) == 1: 
        accel_x, accel_y, accel_z = data[:,:,0], data[:,:,1], data[:,:,2]
        features = [accel_x, accel_y, accel_z]
        name = ['accelx', 'accely', 'accelz']
    i = 0
    for x in features:
        m, M, mean, std =  np.min(x), np.max(x), np.average(x), np.std(x)
        print('min, max, mean, and std. dev of %s:' %name[i], m,',' , M,',', mean,',' ,std)
        i += 1
    print('\n')   
    return

##############################################################################
### Normalization: Mapping to (0,1)
def normalize(data):
    normalized_data = np.zeros((data.shape[0], data.shape[1], data.shape[2]))    
    if len(sensor) == 1:
        if (accel_Or_gyro == 'Accel'):
            accel_x, accel_y, accel_z = data[:,:,0], data[:,:,1], data[:,:,2]
            accel_x = (accel_x-np.amin(accel_x))/(np.amax(accel_x)-np.amin(accel_x))
            accel_y = (accel_y-np.amin(accel_y))/(np.amax(accel_y)-np.amin(accel_y))
            accel_z = (accel_z-np.amin(accel_z))/(np.amax(accel_z)-np.amin(accel_z))
            normalized_data[:,:,0], normalized_data[:,:,1], normalized_data[:,:,2] = accel_x, accel_y, accel_z
        
        if (accel_Or_gyro == 'Gyro'):
            gyro_x, gyro_y, gyro_z = data[:,:,0], data[:,:,1], data[:,:,2]
            gyro_x = (gyro_x-np.amin(gyro_x))/(np.amax(gyro_x)-np.amin(gyro_x))
            gyro_y = (gyro_y-np.amin(gyro_y))/(np.amax(gyro_y)-np.amin(gyro_y))
            gyro_z = (gyro_z-np.amin(gyro_z))/(np.amax(gyro_z)-np.amin(gyro_z))                 
            normalized_data[:,:,0], normalized_data[:,:,1], normalized_data[:,:,2] = gyro_x, gyro_y, gyro_z
    return normalized_data

###### Trim raw signals and remove the zero padding 
def trim(F_raw_X, nonF_raw_X): 
    trimed_F_raw_X, trimed_nonF_raw_X = [], []
    for i in range(F_raw_X.shape[0]):
        F_trimed_signals = []
        X = np.trim_zeros(F_raw_X[i,:,0])
        Y = np.trim_zeros(F_raw_X[i,:,1])
        Z = np.trim_zeros(F_raw_X[i,:,2])       
        length = np.max([len(X),len(Y),len(Z)])
        F_trimed_signals = F_raw_X[i,:length,:]            
        # print(F_trimed_signals.shape)       
        trimed_F_raw_X.append(F_trimed_signals)
       
    for i in range(nonF_raw_X.shape[0]):
        nonF_trimed_signals = []
        X = np.trim_zeros(nonF_raw_X[i,:,0])
        Y = np.trim_zeros(nonF_raw_X[i,:,1])
        Z = np.trim_zeros(nonF_raw_X[i,:,2])       
        length = np.max([len(X),len(Y),len(Z)])
        nonF_trimed_signals = nonF_raw_X[i,:length,:]            
        # print(nonF_trimed_signals.shape)       
        trimed_nonF_raw_X.append(nonF_trimed_signals)
    return trimed_F_raw_X, trimed_nonF_raw_X

##############################################################################
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
imu = data.io.IMUdata(True)  ### using io file

sensor = ['left']  ### can be ['right'], ['left'], or ['neck'] for each of the right, left, or neck sensors
accel_Or_gyro = 'Gyro'  ### can be either 'accel' or 'Gyro' for acceleration or angular velocity signals
Fr = 100
Filtering = 'No' ### can be 'lp', 'hp', or 'No' for lowpass-, highpass-, no-filtering

filepath='C:\\Users\\Venus\\Documents\\Dissertation\\' + sensor + '\\'+ accel_Or_gyro + '\\'
                                                                                              
### Segmented data
data, followup_data = imu.segmented_TUG(sensor, accel_Or_gyro, Filtering, frequency=Fr, window_size=3, stride=1)
[F_segmented_X, F_bag_of_subject_no, F_bag_of_report_fall, F_bag_of_no_of_falls, F_bag_of_comp_score, F_bag_of_seg_no, F_bag_of_age, F_bag_of_gender, F_bag_of_weight, F_bag_of_height, F_bag_of_BP, nonF_segmented_X, nonF_bag_of_subject_no, nonF_bag_of_report_fall, nonF_bag_of_no_of_falls, nonF_bag_of_comp_score, nonF_bag_of_seg_no, nonF_bag_of_age, nonF_bag_of_gender, nonF_bag_of_weight, nonF_bag_of_height, nonF_bag_of_BP] = data
[Faller_segmented_X, Faller_bag_of_subject_no, Faller_bag_of_no_of_falls, Faller_bag_of_comp_score, Faller_bag_of_seg_no, Faller_bag_of_age, Faller_bag_of_gender, Faller_bag_of_weight, Faller_bag_of_height, Faller_bag_of_BP, nonFaller_segmented_X, nonFaller_bag_of_subject_no, nonFaller_bag_of_no_of_falls, nonFaller_bag_of_comp_score, nonFaller_bag_of_seg_no, nonFaller_bag_of_age, nonFaller_bag_of_gender, nonFaller_bag_of_weight, nonFaller_bag_of_height, nonFaller_bag_of_BP] = followup_data
### Raw data
raw_data, raw_followup_data = imu.rawsignal_TUG(sensor, accel_Or_gyro, Filtering, frequency=Fr)
[F_raw_X, F_subject_no, F_report_fall, F_no_of_falls, F_composite_score, F_age, F_gender, F_weight, F_height, F_BP, nonF_raw_X, nonF_subject_no, nonF_report_fall, nonF_no_of_falls, nonF_composite_score, nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP] = raw_data
[Faller_raw_X, Faller_subject_no, Faller_no_of_falls, Faller_composite_score, Faller_age, Faller_gender, Faller_weight, Faller_height, Faller_BP, nonFaller_raw_X, nonFaller_subject_no, nonFaller_no_of_falls, nonFaller_composite_score, nonFaller_age, nonFaller_gender, nonFaller_weight, nonFaller_height, nonFaller_BP] = raw_followup_data

### Print statistics of signals
Print_statistics(F_raw_X)
Print_statistics(nonF_raw_X)

######### Zero Centering and scaling 
normalized_F_segmented_X = np.zeros((F_segmented_X.shape[0], F_segmented_X.shape[1], F_segmented_X.shape[2]))
accel_x, accel_y, accel_z = F_segmented_X[:,:,0], F_segmented_X[:,:,1], F_segmented_X[:,:,2]
accel_x = (accel_x -np.mean(accel_x, axis=0))/np.std(accel_x, axis=0)
accel_y = (accel_y -np.mean(accel_y, axis=0))/np.std(accel_y, axis=0)
accel_z = (accel_z -np.mean(accel_z, axis=0))/np.std(accel_z, axis=0)
normalized_F_segmented_X[:,:,0], normalized_F_segmented_X[:,:,1], normalized_F_segmented_X[:,:,2] = accel_x, accel_y, accel_z

normalized_nonF_segmented_X = np.zeros((nonF_segmented_X.shape[0], nonF_segmented_X.shape[1], nonF_segmented_X.shape[2]))
accel_x, accel_y, accel_z = nonF_segmented_X[:,:,0], nonF_segmented_X[:,:,1], nonF_segmented_X[:,:,2]
accel_x = (accel_x -np.mean(accel_x, axis=0))/np.std(accel_x, axis=0)
accel_y = (accel_y -np.mean(accel_y, axis=0))/np.std(accel_y, axis=0)
accel_z = (accel_z -np.mean(accel_z, axis=0))/np.std(accel_z, axis=0)
normalized_nonF_segmented_X[:,:,0], normalized_nonF_segmented_X[:,:,1], normalized_nonF_segmented_X[:,:,2] = accel_x, accel_y, accel_z

########## Normalize raw signal, fallers and non fallers all together
raw_X = np.concatenate((F_raw_X, nonF_raw_X), axis=0)
normalized_raw_X = normalize(raw_X)

normalized_F_raw_X = normalized_raw_X[:F_raw_X.shape[0],:,:]
normalized_nonF_raw_X = normalized_raw_X[F_raw_X.shape[0]:,:,:]

########## Normalize raw signal after followup-- fallers and non fallers all together
raw_X = np.concatenate((Faller_raw_X, nonFaller_raw_X), axis=0)
normalized_raw_X = normalize(raw_X)

normalized_Faller_raw_X = normalized_raw_X[:Faller_raw_X.shape[0],:,:]
normalized_nonFaller_raw_X = normalized_raw_X[Faller_raw_X.shape[0]:,:,:]

########## Normalize segmented signals, fallers and non fallers all together
segmented_X = np.concatenate((F_segmented_X, nonF_segmented_X), axis=0)
normalized_segmented_X = normalize(segmented_X)

normalized_F_segmented_X = normalized_segmented_X[:F_segmented_X.shape[0],:,:]
normalized_nonF_segmented_X = normalized_segmented_X[F_segmented_X.shape[0]:,:,:]

########## Normalize segmented signals after followup-- fallers and non fallers all together
segmented_X = np.concatenate((Faller_segmented_X, nonFaller_segmented_X), axis=0)
normalized_segmented_X = normalize(segmented_X)

normalized_Faller_segmented_X = normalized_segmented_X[:Faller_segmented_X.shape[0],:,:]
normalized_nonFaller_segmented_X = normalized_segmented_X[Faller_segmented_X.shape[0]:,:,:]

#############################################################################################
### Trim the signals and create a list of fallers and non fallers which will have different shape (different length of time steps)
trimed_F_raw_X, trimed_nonF_raw_X = trim(F_raw_X, nonF_raw_X)

##### Normalize trimmed raw signal, fallers and non fallers all together
def normalize_trimed_data(data):
    normalized_data = []  
    
    Min_x, Max_x = [], []
    Min_y, Max_y = [], []
    Min_z, Max_z = [], []
    for i in range(len(data)):
        accel_x, accel_y, accel_z = data[i][:,0], data[i][:,1], data[i][:,2]
        min_x, min_y, min_z = np.amin(accel_x), np.amin(accel_y), np.amin(accel_z)
        max_x, max_y, max_z = np.amax(accel_x), np.amax(accel_y), np.amax(accel_z)
               
        Min_x.append(min_x)
        Min_y.append(min_y)
        Min_z.append(min_z)
        Max_x.append(max_x)
        Max_y.append(max_y)
        Max_z.append(max_z)
    
    Min_x, Max_x = np.asarray(Min_x), np.asarray(Max_x)
    Min_y, Max_y = np.asarray(Min_y), np.asarray(Max_y)
    Min_z, Max_z = np.asarray(Min_z), np.asarray(Max_z)
    
    Minimum_x, Maximum_x = np.min(Min_x), np.max(Max_x)
    Minimum_y, Maximum_y = np.min(Min_y), np.max(Max_y)
    Minimum_z, Maximum_z = np.min(Min_z), np.max(Max_z)

    for i in range(len(data)):
        accel_x, accel_y, accel_z = data[i][:,0], data[i][:,1], data[i][:,2]
        accel_x = (accel_x-Minimum_x)/(Maximum_x-Minimum_x)
        accel_y = (accel_y-Minimum_y)/(Maximum_y-Minimum_y)
        accel_z = (accel_z-Minimum_z)/(Maximum_z-Minimum_z)       
        normalized = np.concatenate((np.expand_dims(accel_x,axis=1), np.expand_dims(accel_y,axis=1), np.expand_dims(accel_z, axis=1)), axis=1)
        normalized_data.append(normalized)
            
    return normalized_data    

trimed_raw_X = np.concatenate((trimed_F_raw_X, trimed_nonF_raw_X), axis=0)
normalized_trimed_raw_X = normalize_trimed_data(trimed_raw_X)
normalized_trimed_F_raw_X = normalized_trimed_raw_X[:len(trimed_F_raw_X)]
normalized_trimed_nonF_raw_X = normalized_trimed_raw_X[len(trimed_F_raw_X):]

##### Standardize trimmed raw signal, fallers and non fallers all together   
def standardize_trimed_data(trimed_raw_X, raw_X):
    standardized_trimed_data, standardized_data = [], []
    
    accel_x, accel_y, accel_z = [], [], []
    for i in range(len(trimed_raw_X)):
        accel_x.append(trimed_raw_X[i][:,0])
        accel_y.append(trimed_raw_X[i][:,1])
        accel_z.append(trimed_raw_X[i][:,2])
        
    accel_x = np.concatenate(accel_x, axis=0)
    accel_y = np.concatenate(accel_y, axis=0)
    accel_z = np.concatenate(accel_z, axis=0)    
    Mean_x, Mean_y, Mean_z = np.mean(accel_x), np.mean(accel_y), np.mean(accel_z)
    Std_x, Std_y, Std_z = np.std(accel_x), np.std(accel_y), np.std(accel_z)
        
    trimmed_accel_x = np.sqrt(len(accel_x))*(accel_x-Mean_x)/Std_x
    trimmed_accel_y = np.sqrt(len(accel_y))*(accel_y-Mean_y)/Std_y
    trimmed_accel_z = np.sqrt(len(accel_z))*(accel_z-Mean_z)/Std_z
    
    s = 0
    for i in range(len(trimed_raw_X)):
        step = trimed_raw_X[i].shape[0]
        ax = trimmed_accel_x[s : s + step ]
        ay = trimmed_accel_y[s : s + step ]
        az = trimmed_accel_z[s : s + step ]
        s += step
        standardized = np.concatenate((np.expand_dims(ax,axis=1), np.expand_dims(ay,axis=1), np.expand_dims(az, axis=1)), axis=1)
        standardized_trimed_data.append(standardized)
    
    accel_x, accel_y, accel_z = [], [], []
    for i in range(len(raw_X)):
        accel_x = raw_X[i][:,0]
        accel_y = raw_X[i][:,1]
        accel_z = raw_X[i][:,2]
        
        accel_x = np.sqrt(len(accel_x))*(accel_x-Mean_x)/Std_x
        accel_y = np.sqrt(len(accel_y))*(accel_y-Mean_y)/Std_y
        accel_z = np.sqrt(len(accel_z))*(accel_z-Mean_z)/Std_z
        data = np.concatenate((np.expand_dims(accel_x,axis=1), np.expand_dims(accel_y,axis=1), np.expand_dims(accel_z, axis=1)), axis=1)    
        standardized_data.append(data)
    
    return standardized_trimed_data, standardized_data

raw_X = np.concatenate((F_raw_X, nonF_raw_X), axis=0)
standardized_trimed_raw_X, standardized_raw_X = standardize_trimed_data(trimed_raw_X, raw_X)

standardized_trimed_F_raw_X = standardized_trimed_raw_X[:len(trimed_F_raw_X)]
standardized_trimed_nonF_raw_X = standardized_trimed_raw_X[len(trimed_F_raw_X):]

standardized_F_raw_X = standardized_raw_X[:len(trimed_F_raw_X)]
standardized_nonF_raw_X = standardized_raw_X[len(trimed_F_raw_X):]

##############################################################################################
#### Save segmented and separated data based on clinisians' labels
np.save(filepath + 'normalized_F_segmented_X.npy', normalized_F_segmented_X)
np.save(filepath + 'normalized_nonF_segmented_X.npy', normalized_nonF_segmented_X)
np.save(filepath + 'normalized_F_raw_X.npy', normalized_F_raw_X)
np.save(filepath + 'normalized_nonF_raw_X.npy', normalized_nonF_raw_X)

np.save(filepath + 'trimed_F_raw_X.npy', trimed_F_raw_X)
np.save(filepath + 'trimed_nonF_raw_X.npy', trimed_nonF_raw_X)
np.save(filepath + 'normalized_trimed_F_raw_X.npy', normalized_trimed_F_raw_X)
np.save(filepath + 'normalized_trimed_nonF_raw_X.npy', normalized_trimed_nonF_raw_X)

np.save(filepath + 'F_segmented_X.npy', F_segmented_X)
np.save(filepath + 'nonF_segmented_X.npy', nonF_segmented_X)
np.save(filepath + 'F_raw_X.npy', F_raw_X)
np.save(filepath + 'nonF_raw_X.npy', nonF_raw_X)

np.save(filepath + 'F_bag_of_subject_no.npy', F_bag_of_subject_no )
np.save(filepath + 'nonF_bag_of_subject_no.npy', nonF_bag_of_subject_no )
np.save(filepath + 'F_subject_no.npy', F_subject_no )
np.save(filepath + 'nonF_subject_no.npy', nonF_subject_no )

np.save(filepath + 'F_bag_of_seg_no.npy', F_bag_of_seg_no)
np.save(filepath + 'nonF_bag_of_seg_no.npy', nonF_bag_of_seg_no)

np.save(filepath + 'F_bag_of_age.npy', F_bag_of_age)
np.save(filepath + 'nonF_bag_of_age.npy', nonF_bag_of_age)
np.save(filepath + 'F_age.npy', F_age)
np.save(filepath + 'nonF_age.npy', nonF_age)

np.save(filepath + 'F_bag_of_gender.npy', F_bag_of_gender)
np.save(filepath + 'nonF_bag_of_gender.npy', nonF_bag_of_gender)
np.save(filepath + 'F_gender.npy', F_gender)
np.save(filepath + 'nonF_gender.npy', nonF_gender)

np.save(filepath + 'F_bag_of_weight.npy', F_bag_of_weight)
np.save(filepath + 'nonF_bag_of_weight.npy', nonF_bag_of_weight)
np.save(filepath + 'F_weight.npy', F_weight)
np.save(filepath + 'nonF_weight.npy', nonF_weight)

np.save(filepath + 'F_bag_of_height.npy', F_bag_of_height)
np.save(filepath + 'nonF_bag_of_height.npy', nonF_bag_of_height)
np.save(filepath + 'F_height.npy', F_height)
np.save(filepath + 'nonF_height.npy', nonF_height)

np.save(filepath + 'F_bag_of_BP.npy', F_bag_of_BP)
np.save(filepath + 'nonF_bag_of_BP.npy', nonF_bag_of_BP)
np.save(filepath + 'F_BP.npy', F_BP)
np.save(filepath + 'nonF_BP.npy', nonF_BP)