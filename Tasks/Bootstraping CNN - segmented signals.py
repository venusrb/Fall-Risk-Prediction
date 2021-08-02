# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:21:41 2020

@author: Venus
"""

import time
import os
import numpy as np
import keras
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
########################################################
def Sensitivity_And_specificity(cnf_matrix):    
    tn, fp, fn, tp = cnf_matrix[0,0], cnf_matrix[0,1], cnf_matrix[1,0], cnf_matrix[1,1]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tn+ fp+ fn+ tp)    
    return sensitivity, specificity, tp, fp, tn, fn, accuracy
#########################################################################################################
def CNN_model(X):    
    input_layer = keras.layers.Input(shape=(X.shape[1], X.shape[2]), name='input_vals')
    conv_1 = keras.layers.Conv1D(64, kernel_size=5, strides=3,  padding = 'same', data_format = 'channels_last',
                                    name='conv1')(input_layer)
    conv_1 = keras.layers.BatchNormalization(name='conv1_Batch')(conv_1)
    conv_1 = keras.layers.ReLU(name='conv1_relu')(conv_1)
    conv_1 = keras.layers.Dropout(0.2)(conv_1)

    conv_2 = keras.layers.Conv1D(128, kernel_size=5, strides=3, padding = 'same', data_format = 'channels_last',
                                    name='conv2')(conv_1)
    conv_2 = keras.layers.BatchNormalization(name='conv2_Batch')(conv_2)
    conv_2 = keras.layers.ReLU(name='conv2_relu')(conv_2)
    conv_2 = keras.layers.MaxPooling1D(pool_size=2, strides=2, name='conv2_pool')(conv_2)
    conv_2 = keras.layers.Dropout(0.2)(conv_2)
    
    conv_3 = keras.layers.Conv1D(256, kernel_size=3, strides=3, padding = 'same', data_format = 'channels_last',
                                    name='conv3')(conv_2)
    conv_3 = keras.layers.BatchNormalization(name='conv3_Batch')(conv_3)
    conv_3 = keras.layers.ReLU(name='conv3_relu')(conv_3)
    conv_3 = keras.layers.Dropout(0.2)(conv_3)
    
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
########################################################################################################
#############################  LOAD DATA - ACCELEARATION  ##############################################
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\'+'Left\\Accel\\'

### Load the Segemented Data - acceleration
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

##### Load the Normalized Segmented Signals - acceleration
normalized_F_accel = np.load(filepath + 'normalized_F_segmented_X.npy')
normalized_nonF_accel = np.load(filepath + 'normalized_nonF_segmented_X.npy')

print("normalized_F_accel.shape: ", normalized_F_accel.shape)
print("normalized_nonF_accel.shape: ", normalized_nonF_accel.shape)

F_subject_no = np.load(filepath+'F_bag_of_subject_no.npy', allow_pickle=True)
nonF_subject_no = np.load(filepath+'nonF_bag_of_subject_no.npy', allow_pickle=True)
F_seg_no = np.load(filepath+'F_bag_of_seg_no.npy', allow_pickle=True)
nonF_seg_no = np.load(filepath+'nonF_bag_of_seg_no.npy', allow_pickle=True)

F_Y = np.ones((len(normalized_F_accel),1))
nonF_Y = np.zeros((len(normalized_nonF_accel),1))

F_bio = np.concatenate((F_age, F_gender, F_weight, F_height, F_BP), axis=1)
nonF_bio = np.concatenate((nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP), axis=1)
print("Number of Segmented Signal data: ", F_subject_no.shape, nonF_subject_no.shape)

########################################################################################################
############# Create a dictionary of patients with the numbers of their segments
##### Fallers
F_num_seg_dict = {}
F_Subjects = np.sort(np.unique(F_subject_no))
# print(Subjects)
for subject in F_Subjects:
    for i in range(len(F_subject_no)):   
        if subject == F_subject_no[i]:
            subject = int(subject)
            F_num_seg_dict[subject] = F_num_seg_dict.get(subject,0) + 1

##### nonFallers
nonF_num_seg_dict = {}
nonF_Subjects = np.sort(np.unique(nonF_subject_no))

for subject in nonF_Subjects:
    for i in range(len(nonF_subject_no)):   
        if subject == nonF_subject_no[i]:
            subject = int(subject)
            nonF_num_seg_dict[subject] = nonF_num_seg_dict.get(subject,0) + 1

########################################################################################################
#############################  LOAD DATA - GYROSCOPE  ##############################################
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\'+'Left\\Gyro\\'
##### Load the Normalized Segmented Signals - gyroscope
normalized_F_gyro = np.load(filepath + 'normalized_F_segmented_X.npy')
normalized_nonF_gyro = np.load(filepath + 'normalized_nonF_segmented_X.npy')

print("Size of data after adding the reconstructed X: ", normalized_F_accel.shape, normalized_nonF_accel.shape)
print(F_subject_no.shape, nonF_subject_no.shape)
######################

##################################################################################################################
########################################################################### 
def Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro, F_Y, nonF_Y):
   
    ### Create test set
    F_patient_label_test, nonF_patient_label_test = [], []
    F_seg_no_test, nonF_seg_no_test = [], []
    F_accel_test, F_gyro_test, F_bio_test = [], [], []
    nonF_accel_test, nonF_gyro_test, nonF_bio_test = [], [], []
    F_Y_test, nonF_Y_test = [], []
           
    for j in test_F_patients:
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_patient_label_test.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_test.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_test.append(normalized_F_accel[i:i+1,:])
                F_gyro_test.append(normalized_F_gyro[i:i+1,:])
                F_bio_test.append(F_bio[i:i+1,:])
                F_Y_test.append(F_Y[i:i+1,:])
                
    for j in test_nonF_patients:
        for i in range(len(nonF_subject_no)):        
            if (nonF_subject_no[i]==j):
                nonF_patient_label_test.append(np.asarray(nonF_subject_no[i]).reshape(1,1)) 
                nonF_seg_no_test.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_test.append(normalized_nonF_accel[i:i+1,:])
                nonF_gyro_test.append(normalized_nonF_gyro[i:i+1,:])
                nonF_bio_test.append(nonF_bio[i:i+1,:])
                nonF_Y_test.append(nonF_Y[i:i+1,:])

    F_patient_label_test = np.concatenate(F_patient_label_test, axis=0)
    F_seg_no_test = np.concatenate(F_seg_no_test, axis=0)     
    F_accel_test = np.concatenate(F_accel_test, axis=0)
    F_gyro_test = np.concatenate(F_gyro_test, axis=0)    
    F_X_test = np.concatenate((F_accel_test, F_gyro_test), axis=2)

    F_bio_test = np.concatenate(F_bio_test, axis=0)
    F_Y_test = np.concatenate(F_Y_test, axis=0)  
    nonF_patient_label_test = np.concatenate(nonF_patient_label_test, axis=0)  
    nonF_seg_no_test = np.concatenate(nonF_seg_no_test, axis=0)  
    nonF_accel_test = np.concatenate(nonF_accel_test, axis=0)
    nonF_gyro_test = np.concatenate(nonF_gyro_test, axis=0)    
    nonF_X_test = np.concatenate((nonF_accel_test, nonF_gyro_test), axis=2)
    nonF_bio_test = np.concatenate(nonF_bio_test, axis=0)
    nonF_Y_test = np.concatenate(nonF_Y_test, axis=0)
      
    ##### Create test set
    patient_label_test = np.concatenate((F_patient_label_test, nonF_patient_label_test), axis =0)
    seg_no_test = np.concatenate((F_seg_no_test, nonF_seg_no_test), axis =0)
    X_test = np.concatenate((F_X_test, nonF_X_test), axis =0)
    bio_test = np.concatenate((F_bio_test, nonF_bio_test), axis =0)    
    Y_test = np.concatenate((F_Y_test, nonF_Y_test), axis =0)
    patient_label_test = np.squeeze(patient_label_test)
       
    F_test_number = F_X_test.shape[0]
    nonF_test_number = nonF_X_test.shape[0]
    test_data = [patient_label_test, seg_no_test, X_test, F_test_number, nonF_test_number, bio_test, Y_test]    

    return test_data
    
##########################################################################################     
def training(test_data, training_F_patients, training_nonF_patients, val_F_patients, val_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro, F_Y, nonF_Y):
    [patient_label_test, seg_no_test, X_test, F_test_number, nonF_test_number, bio_test, Y_test] = test_data
               
    CV_X_train, CV_Y_train= [], []
    CV_patient_label_train= []
    train_score, test_score= [], []
    CV_test_probabilities =[]
    train_sensitivity, train_specificity, train_AUC = [], [], []
    test_sensitivity, test_specificity, test_AUC = [], [], []     
    
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
    ############################################################################################################
    ##### Only acceleration
    # X_train = X_train[:,:,:3]
    # X_val = X_val[:,:,:3]
    # X_test = X_test[:,:,:3]
       
    ##### Only Gyro
    X_train = X_train[:,:,3:]
    X_val = X_val[:,:,3:]
    X_test = X_test[:,:,3:]
    ############################################################################################################                
    ##### Train and Predict CNN
    adam = keras.optimizers.Adam(lr=0.01)
    model = CNN_model(X_train)
    model.summary()
    
    #### Compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    train_model_name = 'exp_001'
    weight_dir = 'C:\\Users\\Venus\\Documents\\fall-injury\\example\\01_read_csv\\keras-vis-master\\' + 'best_weights_checkpoints'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weight_dir,train_model_name),
                                                  monitor = 'val_loss', verbose=1,  save_best_only=True, mode='min')
    #### Train 3 sec cut model 
    model.fit(X_train, Y_train, epochs=1, batch_size=128, validation_data=[X_val,Y_val], callbacks=[checkpointer])

    #### Load the best model found
    model.load_weights(os.path.join(weight_dir,train_model_name))
        
    #### Evaluate the model
    train_scores = model.evaluate(X_train, Y_train, verbose=0) 
    val_scores = model.evaluate(X_val, Y_val, verbose=0)    
    test_scores = model.evaluate(X_test, Y_test, verbose=0)        
    print("Training:", train_scores) 
    print("Validation:", val_scores) 
    print("Test:", test_scores) 

    train_score = train_scores[1]*100
    test_score = test_scores[1]*100
        
    #### Predict
    train_probabilities = model.predict(X_train)
    train_probabilities = train_probabilities.reshape(train_probabilities.shape[0],1)
    Y_pred_train = np.insert(train_probabilities, 0, 0.5, axis =1)
    Y_pred_train = np.argmax(Y_pred_train, axis=1)
    Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0], 1) 
  
    val_probabilities = model.predict(X_val)
    val_probabilities = val_probabilities.reshape(val_probabilities.shape[0],1)
    Y_pred_val = np.insert(val_probabilities, 0, 0.5, axis =1)
    Y_pred_val = np.argmax(Y_pred_val, axis=1)
    Y_pred_val = Y_pred_val.reshape(Y_pred_val.shape[0], 1)  
                         
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
    train_sensitivity, train_specificity, train_tp, train_fp, train_tn, train_fn, train_accuracy = Sensitivity_And_specificity(train_cnf_matrix) 
    # print("train_specificity", train_specificity)
    
    test_cnf_matrix = confusion_matrix(Y_test, Y_pred_test) # A matrix of truth on row and prediction on column
    test_sensitivity, test_specificity, test_tp, test_fp, test_tn, test_fn, test_accuracy = Sensitivity_And_specificity(test_cnf_matrix)   

    ### Train prediction metrics
    train_sensitivity = train_sensitivity*100
    train_specificity = train_specificity*100        
    train_fpr, train_tpr, train_thresholds = roc_curve(Y_train, train_probabilities)
    train_AUC = auc(train_fpr, train_tpr)*100
    
    ### Test prediction metrics
    test_sensitivity = test_sensitivity*100
    test_specificity = test_specificity*100       
    test_fpr, test_tpr, test_seg_thresholds = roc_curve(Y_test, test_probabilities)
    test_ROC = [test_fpr, test_tpr]
    test_AUC = auc(test_fpr, test_tpr)*100  
    
    test_segment_results = [CV_test_probabilities, test_cnf_matrix, test_score, test_ROC, test_AUC, test_sensitivity, test_specificity]
    train_segment_results = [train_score, train_AUC, train_sensitivity, train_specificity]

    #### Find the average prediction results over the entire segments for each subject in the test set
    Y_Pred_Test = []
    Test_probabilities = []
    Y_Test = []
    for i in np.unique(patient_label_test):
        Y_Pred_Test_subject = []
        Test_probabilities_subject = []
        Y_Test_subject = []
        for j in range(len(patient_label_test)):
            if i==patient_label_test[j]:
                Y_Pred_Test_subject.append(Y_pred_test[j])
                Test_probabilities_subject.append(test_probabilities[j])
                Y_Test_subject.append(Y_test[j])
      
        Y_Pred_Test_subject = np.concatenate(Y_Pred_Test_subject)
        Test_probabilities_subject = np.concatenate(Test_probabilities_subject)
        Y_Test_subject = np.asarray(Y_Test_subject)        
      
        #### Build the arrays of only the voted labels using np.argmax(np.bincount(array)))
        Y_Pred_Test.append(np.argmax(np.bincount(Y_Pred_Test_subject)))
        Test_probabilities.append(np.average(Test_probabilities_subject))
        Y_Test.append(Y_Test_subject[0,0])
 
    Y_Test = np.asarray(Y_Test).reshape(len(Y_Test),1)
    Y_Pred_Test = np.asarray(Y_Pred_Test).reshape(len(Y_Pred_Test),1)  
    
    Test_cnf_matrix = confusion_matrix(Y_Test, Y_Pred_Test) # A matrix of truth on row and prediction on column
    Test_sensitivity, Test_specificity, Test_tp, Test_fp, Test_tn, Test_fn, Test_accuracy = Sensitivity_And_specificity(Test_cnf_matrix)
    
    Test_accuracy = Test_accuracy*100
    Test_sensitivity = Test_sensitivity*100
    Test_specificity = Test_specificity*100 
    Test_fpr, Test_tpr, Test_overall_thresholds = roc_curve(Y_Test, Test_probabilities)
    Test_ROC = [Test_fpr, Test_tpr]
    Test_AUC = auc(Test_fpr, Test_tpr)
    
    Test_Overal_results = [Test_accuracy, Test_ROC, Test_AUC, Test_sensitivity, Test_specificity, Test_probabilities, Test_tp, Test_fp, Test_tn, Test_fn]
    #####################################
      
    data = [X_test, CV_X_train, CV_Y_train, CV_patient_label_train]    
    return model, data, Test_Overal_results, test_segment_results, train_segment_results, test_seg_thresholds, Test_overall_thresholds

#################### Bootstrap ##################### 
t0 = time.time()
bootstrap_iter = 100
test_split = 0.2
                    
boots_train_acc, boots_test_acc, boots_test_ROC, boots_test_AUC, boots_test_sensitivity, boots_test_specificity = [], [], [], [], [], []
boots_Test_tp, boots_Test_fp, boots_Test_tn, boots_Test_fn = [], [], [], []
boots_data, boots_test_patients = [], []
boots_test_seg_thresholds, boots_Test_overall_thresholds = [], []
boots_Test_precision, boots_Test_NPV, boots_Test_F1 = [], [], []
boots_Train_acc, boots_Test_acc, boots_Test_ROC, boots_Test_AUC, boots_Test_sensitivity, boots_Test_specificity = [], [], [], [], [], []
F_test_numbers, nonF_test_numbers = [], []


for i in range(bootstrap_iter):
   
    ### Train, val and test Faller subjects
    train_F_patients, test_F_patients = train_test_split(np.unique(F_subject_no), test_size=0.2)   ### SUBJECT-LEVEL: np.unique to make sure that the train and test set are separated at subject level    
    training_F_patients = train_F_patients[:int(0.75*len(train_F_patients))]
    val_F_patients = train_F_patients[int(0.75*len(train_F_patients)):]
    
    ### Multiplier of the length of resampling of each train, val, and test set
    K = 100
    
    ## Use K as the multiplier
    training_F_patients = np.random.choice(training_F_patients, len(training_F_patients) * K)  ### resampling with replacement
    val_F_patients = np.random.choice(val_F_patients, len(val_F_patients) * K)
    test_F_patients = np.random.choice(test_F_patients, len(test_F_patients) * K)
    
    print("training", len(training_F_patients))
    print("val", len(val_F_patients))
    print("test:", len(test_F_patients))
    
    ###################################################################         
    ### Train, val and test nonFaller subjects
    train_nonF_patients, test_nonF_patients = train_test_split(np.unique(nonF_subject_no), test_size=0.2)   ### SUBJECT-LEVEL: np.unique to make sure that the train and test set are separated at subject level    
    training_nonF_patients = train_nonF_patients[:int(0.75*len(train_nonF_patients))]
    val_nonF_patients = train_nonF_patients[int(0.75*len(train_nonF_patients)):]
    
    ## Use K as the multiplier
    training_nonF_patients = np.random.choice(training_nonF_patients, len(training_nonF_patients) * K)
    val_nonF_patients = np.random.choice(val_nonF_patients, len(val_nonF_patients) * K)
    test_nonF_patients = np.random.choice(test_nonF_patients, len(test_nonF_patients)* K)
      
    print("training: ", len(training_nonF_patients))
    print("test: ", len(test_nonF_patients))
    print("val: ", len(val_nonF_patients))
   

    test_data = Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro, F_Y, nonF_Y)    
    [patient_label_test, seg_no_test, X_test, F_test_number, nonF_test_number, bio_test, Y_test] = test_data
    F_test_numbers.append(F_test_number)
    nonF_test_numbers.append(nonF_test_number)

      
    model, data, Test_Overal_results, test_segment_results, train_segment_results, test_seg_thresholds, Test_overall_thresholds = training(test_data, training_F_patients, training_nonF_patients, val_F_patients, val_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro, F_Y, nonF_Y)
    [X_test, X_train, Y_train, patient_label_train] = data
    boots_data.append(data)
    
    [CV_test_probabilities, test_cnf_matrix, test_score, test_ROC, test_AUC, test_sensitivity, test_specificity] = test_segment_results
    [train_score, train_AUC, train_sensitivity, train_specificity] = train_segment_results
    
    [Test_accuracy, Test_ROC, Test_AUC, Test_sensitivity, Test_specificity, Test_probabilities, Test_tp, Test_fp, Test_tn, Test_fn] = Test_Overal_results
    [Test_fpr, Test_tpr] = Test_ROC
    
    
    if Test_tn==0 and Test_fn==0:
        Test_NPV = 0
    else:
        Test_NPV = Test_tn/(Test_tn+Test_fn)
    if Test_tp==0 and Test_fp==0:
        Test_precision = 0
    else:
        Test_precision = Test_tp/(Test_tp+Test_fp)
    if Test_sensitivity == 0 and Test_precision == 0:
        Test_F1 = 0
    else:
        Test_F1 = (2*(Test_sensitivity/100)*Test_precision) / ((Test_sensitivity/100)+Test_precision)
    
    boots_Test_tp.append(Test_tp)
    boots_Test_fp.append(Test_fp)
    boots_Test_tn.append(Test_tn)
    boots_Test_fn.append(Test_fn)
    
    boots_train_acc.append(train_score)
    boots_test_acc.append(test_score)
    boots_test_sensitivity.append(test_sensitivity)
    boots_test_specificity.append(test_specificity)
    boots_test_AUC.append(test_AUC)
    boots_test_seg_thresholds.append(test_seg_thresholds)
       
    #### Ovral results
    boots_Test_acc.append(Test_accuracy)
    boots_Test_sensitivity.append(Test_sensitivity)
    boots_Test_specificity.append(Test_specificity)
    boots_Test_AUC.append(Test_AUC)
       
    boots_Test_ROC.append(Test_ROC)
    boots_Test_overall_thresholds.append(Test_overall_thresholds)
            
    boots_Test_NPV.append(Test_NPV)
    boots_Test_precision.append(Test_precision)
    boots_Test_F1.append(Test_F1)


print('\n')
print("Overall Prediction Results") 
print("acc", boots_Test_acc)
print("sensitivity", boots_Test_sensitivity)
print("specificity", boots_Test_specificity)
print("AUC", boots_Test_AUC)
print('---------------------------------------------------------------')
print("ROC", boots_Test_ROC)
print("thresholds", boots_Test_overall_thresholds)
print('---------------------------------------------------------------')
print("tp", boots_Test_tp)
print("fp", boots_Test_fp)
print("tn", boots_Test_tn)
print("fn", boots_Test_fn)
print('---------------------------------------------------------------')
print('\n')
print("precision", boots_Test_precision)
print("NPV", boots_Test_NPV)
print("F1", boots_Test_F1)

print('\n')   
print("Acc: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_acc), np.percentile(boots_Test_acc, 2.5),np.percentile(boots_Test_acc, 97.5))) 
print("SE: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_sensitivity), np.percentile(boots_Test_sensitivity, 2.5),np.percentile(boots_Test_sensitivity, 97.5)))
print("SP: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_specificity), np.percentile(boots_Test_specificity, 2.5),np.percentile(boots_Test_specificity, 97.5)))
print("AUC: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_AUC), np.percentile(boots_Test_AUC, 2.5),np.percentile(boots_Test_AUC, 97.5)))
print('---------------------------------------------------------------')
print("NPV: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_NPV), np.percentile(boots_Test_NPV, 2.5),np.percentile(boots_Test_NPV, 97.5)))
print("Precision: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_precision), np.percentile(boots_Test_precision, 2.5),np.percentile(boots_Test_precision, 97.5)))
print("F1: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_F1), np.percentile(boots_Test_F1, 2.5),np.percentile(boots_Test_F1, 97.5)))
print('---------------------------------------------------------------')
print("TP: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_tp), np.percentile(boots_Test_tp, 2.5),np.percentile(boots_Test_tp, 97.5)))
print("FP: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_fp), np.percentile(boots_Test_fp, 2.5),np.percentile(boots_Test_fp, 97.5)))
print("TN: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_tn), np.percentile(boots_Test_tn, 2.5),np.percentile(boots_Test_tn, 97.5)))
print("FN: %.2f%% (%.2f%%, %.2f%%)" % (np.mean(boots_Test_fn), np.percentile(boots_Test_fn, 2.5),np.percentile(boots_Test_fn, 97.5))) 
print('\n')  
###################################################################
avg_F_test_numbers, avg_nonF_test_numbers = np.average(np.asarray(F_test_numbers)), np.average(np.asarray(nonF_test_numbers))  
print("avg_F_test_numbers, avg_nonF_test_numbers:   ", avg_F_test_numbers, avg_nonF_test_numbers) 
print("F_test_numbers:  ", F_test_numbers)
print("nonF_test_numbers:  ", nonF_test_numbers)
print('\n')

#### Print the Computation Time
t1 = time.time()
print("running time (hours):", (t1-t0)/3600)