
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:27:56 2020

@author: venusroshdi
"""
import time
import os
import numpy as np
import keras
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations

##################################################
def Sensitivity_And_specificity(cnf_matrix):    
    tn, fp, fn, tp = cnf_matrix[0,0], cnf_matrix[0,1], cnf_matrix[1,0], cnf_matrix[1,1]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tn+ fp+ fn+ tp)    
    return sensitivity, specificity, tp, fp, accuracy

###########################################################
def CNN_model(X):    
    input_layer = keras.layers.Input(shape=(X.shape[1], X.shape[2]), name='input_vals')
    conv_1 = keras.layers.Conv1D(64, kernel_size=5, strides=3,  padding = 'same', data_format = 'channels_last',
                                    name='conv1')(input_layer)
    conv_1 = keras.layers.BatchNormalization(name='conv1_Batch')(conv_1)
    conv_1 = keras.layers.ReLU(name='conv1_relu')(conv_1)

    conv_2 = keras.layers.Conv1D(128, kernel_size=5, strides=3, padding = 'same', data_format = 'channels_last',
                                    name='conv2')(conv_1)
    conv_2 = keras.layers.BatchNormalization(name='conv2_Batch')(conv_2)
    conv_2 = keras.layers.ReLU(name='conv2_relu')(conv_2)
    conv_2 = keras.layers.MaxPooling1D(pool_size=2, strides=2, name='conv2_pool')(conv_2)
    
    conv_3 = keras.layers.Conv1D(256, kernel_size=3, strides=3, padding = 'same', data_format = 'channels_last',
                                    name='conv3')(conv_2)
    conv_3 = keras.layers.BatchNormalization(name='conv3_Batch')(conv_3)
    conv_3 = keras.layers.ReLU(name='conv3_relu')(conv_3)
    
    conv_4 = keras.layers.Conv1D(128, kernel_size=3, strides=3, padding = 'same', data_format = 'channels_last',
                                    name='conv4')(conv_3)
    conv_4 = keras.layers.BatchNormalization(name='conv4_Batch')(conv_4)
    conv_4 = keras.layers.ReLU(name='conv4_relu')(conv_4)
    conv_4 = keras.layers.MaxPooling1D(pool_size=2, strides=2, name='conv4_pool')(conv_4)
    
    flatten = keras.layers.Flatten(name='flatten')(conv_4)
    fc_1 = keras.layers.Dense(1,name='fc_1')(flatten)
    fc_1 = keras.layers.BatchNormalization(name='fc1_Batch')(fc_1) 
    
    output_layer = keras.layers.Activation(activation='sigmoid', name='output_layer_activation')(fc_1)        
    model = keras.Model(inputs=input_layer, outputs=output_layer)     
    return model
###########################################################################
filepath='C:\\Users\\Venus\\Documents\\fall-injury\\example\\01_read_csv\\keras-vis-master\\'+'100Hz-Mapped-All-together\\Left\\Accel\\' 

F_age = np.load(filepath+'F_bag_of_age.npy')
N = len(F_age)
F_age = F_age.reshape(N,1)
F_gender = np.load(filepath+'F_bag_of_gender.npy').reshape(N,1)
F_weight = np.load(filepath+'F_bag_of_weight.npy').reshape(N,1)
F_height= np.load(filepath+'F_bag_of_height.npy').reshape(N,1)
F_BP = np.load(filepath+'F_bag_of_BP.npy').reshape(N,1)

nonF_age = np.load(filepath+'nonF_bag_of_age.npy')
M = len(nonF_age)
nonF_age = nonF_age.reshape(M,1)
nonF_gender = np.load(filepath+'nonF_bag_of_gender.npy').reshape(M,1)
nonF_weight = np.load(filepath+'nonF_bag_of_weight.npy').reshape(M,1)
nonF_height= np.load(filepath+'nonF_bag_of_height.npy').reshape(M,1)
nonF_BP = np.load(filepath+'nonF_bag_of_BP.npy').reshape(M,1)

normalized_F_accel = np.load(filepath + 'normalized_F_segmented_X.npy')
normalized_nonF_accel = np.load(filepath + 'normalized_nonF_segmented_X.npy')
# print(np.min(normalized_F_accel), np.min(normalized_nonF_accel))
# print(np.max(normalized_F_accel), np.max(normalized_nonF_accel))

F_subject_no = np.load(filepath+'F_bag_of_subject_no.npy', allow_pickle=True)
nonF_subject_no = np.load(filepath+'nonF_bag_of_subject_no.npy', allow_pickle=True)
F_seg_no = np.load(filepath+'F_bag_of_seg_no.npy', allow_pickle=True)
nonF_seg_no = np.load(filepath+'nonF_bag_of_seg_no.npy', allow_pickle=True)

F_bio = np.concatenate((F_age, F_gender, F_weight, F_height, F_BP), axis=1)
nonF_bio = np.concatenate((nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP), axis=1)

### Load raw data and normalize it for each subject separately
def normalize_subjects_separately(data):
    normalized_data = []
    for i in range(len(data)):
        accel_x, accel_y, accel_z = data[i][:,0], data[i][:,1], data[i][:,2]
        accel_x = (accel_x-np.amin(accel_x))/(np.amax(accel_x)-np.amin(accel_x))
        accel_y = (accel_y-np.amin(accel_y))/(np.amax(accel_y)-np.amin(accel_y))
        accel_z = (accel_z-np.amin(accel_z))/(np.amax(accel_z)-np.amin(accel_z))
        Accel = np.concatenate((np.expand_dims(accel_x,axis=1), np.expand_dims(accel_y,axis=1), np.expand_dims(accel_z,axis=1)), axis=1)        
        normalized_data.append(Accel)
    return normalized_data
############################
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\Left\\Gyro\\' 
normalized_F_gyro = np.load(filepath + 'normalized_F_segmented_X.npy')
normalized_nonF_gyro = np.load(filepath + 'normalized_nonF_segmented_X.npy')

########################################################################### 
def Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro):
    ### Create test set
    F_patient_label_test, nonF_patient_label_test = [], []
    F_seg_no_test, nonF_seg_no_test = [], []
    F_accel_test, F_gyro_test, F_bio_test = [], [], []
    nonF_accel_test, nonF_gyro_test, nonF_bio_test = [], [], []
        
    # print('F_patient_test_list', len(F_patient_test_list)) # print('nonF_patient_test_list', len(nonF_patient_test_list))   
    
    for j in test_F_patients:
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_patient_label_test.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_test.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_test.append(normalized_F_accel[i:i+1,:])
                F_gyro_test.append(normalized_F_gyro[i:i+1,:])
                F_bio_test.append(F_bio[i:i+1,:])

                
    for j in test_nonF_patients:
        for i in range(len(nonF_subject_no)):        
            if (nonF_subject_no[i]==j):
                nonF_patient_label_test.append(np.asarray(nonF_subject_no[i]).reshape(1,1)) 
                nonF_seg_no_test.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_test.append(normalized_nonF_accel[i:i+1,:])
                nonF_gyro_test.append(normalized_nonF_gyro[i:i+1,:])
                nonF_bio_test.append(nonF_bio[i:i+1,:])
  

    F_patient_label_test = np.concatenate(F_patient_label_test, axis=0)
    F_seg_no_test = np.concatenate(F_seg_no_test, axis=0)     
    F_accel_test = np.concatenate(F_accel_test, axis=0)
    F_gyro_test = np.concatenate(F_gyro_test, axis=0)    
    F_X_test = np.concatenate((F_accel_test, F_gyro_test), axis=2)
    F_bio_test = np.concatenate(F_bio_test, axis=0)  

    nonF_patient_label_test = np.concatenate(nonF_patient_label_test, axis=0)  
    nonF_seg_no_test = np.concatenate(nonF_seg_no_test, axis=0)  
    nonF_accel_test = np.concatenate(nonF_accel_test, axis=0)
    nonF_gyro_test = np.concatenate(nonF_gyro_test, axis=0)    
    nonF_X_test = np.concatenate((nonF_accel_test, nonF_gyro_test), axis=2)
    nonF_bio_test = np.concatenate(nonF_bio_test, axis=0)
      
    ##### Create test set
    test_faller_y = np.ones((len(F_bio_test),1))
    test_nonfaller_y = np.zeros((len(nonF_bio_test),1))

    patient_label_test = np.concatenate((F_patient_label_test, nonF_patient_label_test), axis =0)
    seg_no_test = np.concatenate((F_seg_no_test, nonF_seg_no_test), axis =0)
    X_test = np.concatenate((F_X_test, nonF_X_test), axis =0)
    bio_test = np.concatenate((F_bio_test, nonF_bio_test), axis =0)
    
    Y_test = np.concatenate((test_faller_y, test_nonfaller_y), axis =0)
    patient_label_test = np.squeeze(patient_label_test)
    
    print("X_test", X_test.shape)

    test_data = [patient_label_test, seg_no_test, X_test, bio_test, Y_test]
    return test_data
    
##########################################         10-fold Cross-Validation         ##########################################
def training(test_data, training_F_patients, training_nonF_patients, val_F_patients, val_nonF_patients, F_bio, nonF_bio, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro):
    [patient_label_test, seg_no_test, X_test, bio_test, Y_test] = test_data
               
    CV_X_train, CV_Y_train= [], []
    CV_patient_label_train= []
    train_score, test_score= [], []
    CV_test_probabilities =[]
    train_sensitivity, train_specificity, train_AUC = [], [], []
    test_sensitivity, test_specificity, test_AUC = [], [], []     
    
    ######################
    F_Y = np.ones((len(F_subject_no),1))
    nonF_Y = np.zeros((len(nonF_subject_no),1))
    
    ### Separate training patients
    F_Y_train, F_patient_label_train, F_seg_no_train = [], [], []
    F_accel_train, F_gyro_train, F_bio_train = [], [], []

    nonF_Y_train, nonF_patient_label_train, nonF_seg_no_train = [], [], []
    nonF_accel_train, nonF_gyro_train, nonF_bio_train = [], [], []

    for j in training_F_patients: 
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_Y_train.append(F_Y[i])
                F_patient_label_train.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_train.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_train.append(np.asarray(normalized_F_accel[i:i+1,:]))
                F_gyro_train.append(np.asarray(normalized_F_gyro[i:i+1,:]))
                F_bio_train.append(np.asarray(F_bio[i:i+1,:]))
          
    for j in training_nonF_patients:
        for i in range(len(nonF_subject_no)):    
            if (nonF_subject_no[i]==j):    
                nonF_Y_train.append(nonF_Y[i])
                nonF_patient_label_train.append(np.asarray(nonF_subject_no[i]).reshape(1,1))
                nonF_seg_no_train.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_train.append(np.asarray(normalized_nonF_accel[i:i+1,:]))
                nonF_gyro_train.append(np.asarray(normalized_nonF_gyro[i:i+1,:]))
                nonF_bio_train.append(np.asarray(nonF_bio[i:i+1,:]))

    F_Y_train = np.concatenate(F_Y_train, axis=0)
    F_patient_label_train = np.concatenate(F_patient_label_train, axis=0)
    F_seg_no_train = np.concatenate(F_seg_no_train, axis=0)
    F_accel_train = np.concatenate(F_accel_train, axis=0)
    F_gyro_train = np.concatenate(F_gyro_train, axis=0)
    F_X_train = np.concatenate((F_accel_train, F_gyro_train), axis=2)
    F_bio_train = np.concatenate(F_bio_train, axis=0)

    nonF_Y_train = np.concatenate(nonF_Y_train, axis=0)
    nonF_patient_label_train = np.concatenate(nonF_patient_label_train, axis=0)
    nonF_seg_no_train = np.concatenate(nonF_seg_no_train, axis=0)
    nonF_accel_train = np.concatenate(nonF_accel_train, axis=0)
    nonF_gyro_train = np.concatenate(nonF_gyro_train, axis=0)
    nonF_X_train = np.concatenate((nonF_accel_train, nonF_gyro_train), axis=2)
    nonF_bio_train = np.concatenate(nonF_bio_train, axis=0)

    ### Separate val patients
    F_Y_val, F_patient_label_val, F_seg_no_val = [], [], []
    F_accel_val, F_gyro_val, F_bio_val = [], [], []

    nonF_Y_val, nonF_patient_label_val, nonF_seg_no_val = [], [], []
    nonF_accel_val, nonF_gyro_val, nonF_bio_val = [], [], []
    
    for j in val_F_patients: 
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_Y_val.append(F_Y[i])
                F_patient_label_val.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_val.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_val.append(np.asarray(normalized_F_accel[i:i+1,:]))
                F_gyro_val.append(np.asarray(normalized_F_gyro[i:i+1,:]))
                F_bio_val.append(np.asarray(F_bio[i:i+1,:]))
          
    for j in val_nonF_patients:
        for i in range(len(nonF_subject_no)):    
            if (nonF_subject_no[i]==j):    
                nonF_Y_val.append(nonF_Y[i])
                nonF_patient_label_val.append(np.asarray(nonF_subject_no[i]).reshape(1,1))
                nonF_seg_no_val.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_val.append(np.asarray(normalized_nonF_accel[i:i+1,:]))
                nonF_gyro_val.append(np.asarray(normalized_nonF_gyro[i:i+1,:]))
                nonF_bio_val.append(np.asarray(nonF_bio[i:i+1,:]))
   
    F_Y_val = np.concatenate(F_Y_val, axis=0)
    F_patient_label_val = np.concatenate(F_patient_label_val, axis=0)
    F_seg_no_val = np.concatenate(F_seg_no_val, axis=0)
    F_accel_val = np.concatenate(F_accel_val, axis=0)
    F_gyro_val = np.concatenate(F_gyro_val, axis=0)
    F_X_val = np.concatenate((F_accel_val, F_gyro_val), axis=2)
    F_bio_val = np.concatenate(F_bio_val, axis=0)

    nonF_Y_val = np.concatenate(nonF_Y_val, axis=0)
    nonF_patient_label_val = np.concatenate(nonF_patient_label_val, axis=0)
    nonF_seg_no_val = np.concatenate(nonF_seg_no_val, axis=0)
    nonF_accel_val = np.concatenate(nonF_accel_val, axis=0)
    nonF_gyro_val = np.concatenate(nonF_gyro_val, axis=0)
    nonF_X_val = np.concatenate((nonF_accel_val, nonF_gyro_val), axis=2)
    nonF_bio_val = np.concatenate(nonF_bio_val, axis=0)

   ######################################
    
    Y_train, patient_label_train, seg_no_train = [], [], []
    Y_val, patient_label_val, seg_no_val = [], [], []
    X_train, bio_train = [], []
    X_val, bio_val = [], []
    
    # Build the training set such that it includes faller and non faller every other
    for i in range(np.maximum(len(F_X_train), len(nonF_X_train))):
        if(i < len(F_X_train)):
            Y_train.append(np.ones(1))
            patient_label_train.append(F_patient_label_train[i])
            seg_no_train.append(F_seg_no_train[i])
            X_train.append(F_X_train[i]) 
            bio_train.append(F_bio_train[i]) 

        if(i < len(nonF_X_train)):  
            Y_train.append(np.zeros(1))
            patient_label_train.append(nonF_patient_label_train[i])
            seg_no_train.append(nonF_seg_no_train[i])
            X_train.append(nonF_X_train[i])
            bio_train.append(nonF_bio_train[i])
  
    Y_train = np.concatenate(Y_train, axis=0)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    
    patient_label_train = np.squeeze(patient_label_train)
    seg_no_train = np.squeeze(seg_no_train)
    X_train = np.squeeze(X_train)
    bio_train = np.squeeze(bio_train)
    
    # Build the val set such that it includes faller and non faller every other
    for i in range(np.maximum(len(F_X_val), len(nonF_X_val))):
        if(i < len(F_X_val)):
            Y_val.append(np.ones(1))
            patient_label_val.append(F_patient_label_val[i])
            seg_no_val.append(F_seg_no_val[i])
            X_val.append(F_X_val[i]) 
            bio_val.append(F_bio_val[i]) 

        if(i < len(nonF_X_val)):  
            Y_val.append(np.zeros(1))
            patient_label_val.append(nonF_patient_label_val[i])
            seg_no_val.append(nonF_seg_no_val[i])
            X_val.append(nonF_X_val[i])
            bio_val.append(nonF_bio_val[i])
  
    Y_val = np.concatenate(Y_val, axis=0)
    Y_val = Y_val.reshape(Y_val.shape[0], 1)
    
    patient_label_val = np.squeeze(patient_label_val)
    seg_no_val = np.squeeze(seg_no_val)
    X_val = np.squeeze(X_val)
    bio_val = np.squeeze(bio_val)
    
    print("X_val", X_val.shape)    
####################################################################################
    #### Only Gyro
    X_train = X_train[:,:,3:]
    X_val = X_val[:,:,3:]
    X_test = X_test[:,:,3:]
#################################################################################                
    ### Train and Predict CNN
    adam = keras.optimizers.Adam(lr=0.01)
    model = CNN_model(X_train)
    model.summary()
    
    ## Compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    train_model_name = 'exp_001'
    weight_dir = 'C:\\Users\\Venus\\Documents\\Dissertation\\' + 'best_weights_checkpoints'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir,train_model_name),
                                                  monitor = 'val_loss', verbose=1,  save_best_only=True, mode='min')
    # Train the model 
    model.fit(X_train, Y_train, epochs=3, batch_size=128, validation_data=[X_val,Y_val], callbacks=[checkpointer])

    #### Load the best model found
    model.load_weights(os.path.join(weight_dir,train_model_name))
    
    
    #### Validate model
    train_probabilities = model.predict(X_train)
    train_probabilities = train_probabilities.reshape(train_probabilities.shape[0],1)
    # CV_train_probabilities.append(train_probabilities)
    Y_pred_train = np.insert(train_probabilities, 0, 0.5, axis =1)
    Y_pred_train = np.argmax(Y_pred_train, axis=1)
    Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0], 1) 
  
    val_probabilities = model.predict(X_val)
    val_probabilities = val_probabilities.reshape(val_probabilities.shape[0],1)
    Y_pred_val = np.insert(val_probabilities, 0, 0.5, axis =1)
    Y_pred_val = np.argmax(Y_pred_val, axis=1)
    Y_pred_val = Y_pred_val.reshape(Y_pred_val.shape[0], 1)  
    
    # evaluate the model
    train_scores = model.evaluate(X_train, Y_train, verbose=0) 
    val_scores = model.evaluate(X_val, Y_val, verbose=0)    
    test_scores = model.evaluate(X_test, Y_test, verbose=0)        
    print("Training:", train_scores) 
    print("Validation:", val_scores) 
    print("Test:", test_scores) 

    train_score = train_scores[1]*100
    test_score = test_scores[1]*100
                      
    test_probabilities = model.predict(X_test)
    test_probabilities = test_probabilities.reshape(test_probabilities.shape[0],1)
    CV_test_probabilities.append(test_probabilities)
    Y_pred_test = np.insert(test_probabilities, 0, 0.5, axis =1)
    Y_pred_test = np.argmax(Y_pred_test, axis=1)
    Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0], 1)    
       
###################################################
    ### Create list of train and val set over 10-fold CV
    CV_X_train.append(X_train)
    CV_Y_train.append(Y_train)
    CV_patient_label_train.append(patient_label_train)
###################################################
    train_cnf_matrix = confusion_matrix(Y_train, Y_pred_train) # A matrix of truth on row and prediction on column
    train_sensitivity, train_specificity, train_tp, train_fp, train_accuracy = Sensitivity_And_specificity(train_cnf_matrix) 
    
    test_cnf_matrix = confusion_matrix(Y_test, Y_pred_test) # A matrix of truth on row and prediction on column
    test_sensitivity, test_specificity, test_tp, test_fp, test_accuracy = Sensitivity_And_specificity(test_cnf_matrix)            

    ### Train prediction metrics
    train_sensitivity = train_sensitivity*100
    train_specificity = train_specificity*100        
    train_fpr, train_tpr, train_thresholds = roc_curve(Y_train, train_probabilities)
    train_AUC = auc(train_fpr, train_tpr)*100
    
    ### Test prediction metrics
    test_sensitivity = test_sensitivity*100
    test_specificity = test_specificity*100       
    test_fpr, test_tpr, test_thresholds = roc_curve(Y_test, test_probabilities)
    test_AUC = auc(test_fpr, test_tpr)*100
  
    data = [X_test, CV_X_train, CV_Y_train, CV_patient_label_train, X_val, Y_val, patient_label_val, val_probabilities]    
    return model, CV_test_probabilities, data, test_cnf_matrix, train_score, train_AUC, test_score, test_AUC, test_sensitivity, test_specificity

###### Bootstrap
t0 = time.time()
bootstrap_iter = 1
test_split = 0.2

bs_train_acc, bs_test_acc, bs_test_AUC, bs_test_sensitivity, bs_test_specificity = [], [], [], [], []
BS_data, BS_test_patients = [], []

for i in range(bootstrap_iter):
    
    ### Train and test subjects
    F_subject_no = np.unique(F_subject_no)
    train_F_patients = F_subject_no[:int(0.8*len(F_subject_no))]
    test_F_patients = F_subject_no[int(0.8*len(F_subject_no)):]
    
    training_F_patients = train_F_patients[:int(0.8*len(train_F_patients))]
    val_F_patients = train_F_patients[int(0.8*len(train_F_patients)):]

    K = 10  ### Multiplier of the length of resampled set
    training_F_patients = np.random.choice(training_F_patients, len(training_F_patients)*K)
    val_F_patients = np.random.choice(val_F_patients, len(val_F_patients)*K)
    test_F_patients = np.random.choice(test_F_patients, len(test_F_patients)*K)

    nonF_subject_no = np.unique(nonF_subject_no)
    train_nonF_patients = nonF_subject_no[:int(0.8*len(nonF_subject_no))]
    test_nonF_patients = nonF_subject_no[int(0.8*len(nonF_subject_no)):]
    
    training_nonF_patients = train_nonF_patients[:int(0.8*len(train_nonF_patients))]
    val_nonF_patients = train_nonF_patients[int(0.8*len(train_nonF_patients)):]
    
    training_nonF_patients = np.random.choice(training_nonF_patients, len(training_nonF_patients)*K)
    val_nonF_patients = np.random.choice(val_nonF_patients, len(val_nonF_patients)*K)
    test_nonF_patients = np.random.choice(test_nonF_patients, len(test_nonF_patients)*K)


    test_data = Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro)
      
    model, CV_test_probabilities, data, test_cnf_matrix, train_score, train_AUC, test_score, test_AUC, test_sensitivity, test_specificity = training(test_data, training_F_patients, training_nonF_patients, val_F_patients, val_nonF_patients, F_bio, nonF_bio, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro)
    [X_test, X_train, Y_train, patient_label_train, X_val, Y_val, patient_label_val, val_probabilities] = data
    
    BS_data.append(data)
    
    bs_train_acc.append(train_score)
    bs_test_acc.append(test_score)
    bs_test_sensitivity.append(test_sensitivity)
    bs_test_specificity.append(test_specificity)
    bs_test_AUC.append(test_AUC)
    
      
print("BS Train Accuracy: %.2f%% (%.2f%%,%.2f%%)" % (np.mean(bs_train_acc), np.percentile(bs_train_acc, 2.5), np.percentile(bs_train_acc, 97.5))) 
print('\n')

print("BS Test Accuracy: %.2f%% (%.2f%%,%.2f%%)" % (np.mean(bs_test_acc), np.percentile(bs_test_acc, 2.5),np.percentile(bs_test_acc, 97.5))) 
print("BS Test Sensitivity: %.2f%% (%.2f%%,%.2f%%)" % (np.mean(bs_test_sensitivity), np.percentile(bs_test_sensitivity, 2.5),np.percentile(bs_test_sensitivity, 97.5)))
print("BS Test Specificity: %.2f%% (%.2f%%,%.2f%%)" % (np.mean(bs_test_specificity), np.percentile(bs_test_specificity, 2.5),np.percentile(bs_test_specificity, 97.5)))
print("BS Test AUC: %.2f%% (%.2f%%,%.2f%%)" % (np.mean(bs_test_AUC), np.percentile(bs_test_AUC, 2.5),np.percentile(bs_test_AUC, 97.5)))

t1 = time.time()
print('\n')
print("running time:", t1-t0)


### Grad CAM Visualization for binary classification
class_idx = 1 # class of interest index. faller=1, non-falller=0
              # In binary classification there is only one nueron output
              # It needs to be the same for both faller and nonfaller visualization
example_class = 1 # class of examples. faller=1, non-falller=0
indices = np.where(Y_val[:] == example_class)[0] # Extract the indices of class of example in val set
print(indices)
# pick some random input from that class of interest. Any random number from 0 to len(indices)-1
#idx = indices[12]

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'fc_1')

### Swap softmax with linear in the model
vis_model = model
vis_model.layers[layer_idx].activation = activations.linear
vis_model = utils.apply_modifications(vis_model)

##grads = visualize_saliency(vis_model, layer_idx, filter_indices=class_idx, seed_input=X_val[idx], backprop_modifier='guided')
## 'guided': Modifies backprop to only propagate positive gradients for positive activations

### For both categorical and binary classification
penultimate_layer = utils.find_layer_idx(model, 'conv1_Batch')
indices = indices[:1] ## Plot only 10 of 78 segments

size = len(indices)
fig, axes = plt.subplots(size,1)
### seaborn heatmap
for i, idx in enumerate(indices):
    print(idx)
    grads = visualize_cam(vis_model, layer_idx, filter_indices=class_idx, seed_input=X_val[idx], penultimate_layer_idx=penultimate_layer,
                          backprop_modifier='relu')

