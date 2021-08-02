# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:21:41 2020

@author: Venus
"""

import time
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

##################################################################################################
#############################  LOAD DATA - ACCELEARATION  ########################################
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\'+'Left\\Accel\\'

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

F_subject_no = np.load(filepath+'F_bag_of_subject_no.npy', allow_pickle=True)
nonF_subject_no = np.load(filepath+'nonF_bag_of_subject_no.npy', allow_pickle=True)
F_seg_no = np.load(filepath+'F_bag_of_seg_no.npy', allow_pickle=True)
nonF_seg_no = np.load(filepath+'nonF_bag_of_seg_no.npy', allow_pickle=True)

F_bio = np.concatenate((F_age, F_gender, F_weight, F_height, F_BP), axis=1)
nonF_bio = np.concatenate((nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP), axis=1)

F_Y = np.ones((len(normalized_F_accel),1))
nonF_Y = np.zeros((len(normalized_nonF_accel),1))

print("Number of Segmented Signal data: ", F_subject_no.shape, nonF_subject_no.shape)
print("F_subject_no", len(np.unique(F_subject_no)), np.unique(F_subject_no))
print("nonF_subject_no", len(np.unique(nonF_subject_no)), np.unique(nonF_subject_no))

###################################################################################################
#############################  LOAD DATA - GYROSCOPE  #############################################
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\'+'Left\\Gyro\\' 

##### Load the Normalized Segmented Signals - gyroscope
normalized_F_gyro = np.load(filepath + 'normalized_F_segmented_X.npy')
normalized_nonF_gyro = np.load(filepath + 'normalized_nonF_segmented_X.npy')


###################################################################################################
def Sensitivity_And_specificity(cnf_matrix):    
    tn, fp, fn, tp = cnf_matrix[0,0], cnf_matrix[0,1], cnf_matrix[1,0], cnf_matrix[1,1]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tn+ fp+ fn+ tp)    
    return sensitivity, specificity, tp, fp, tn, fn, accuracy
####################################################################################################
def signal_statistics(normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro):

    F_accel, F_gyro, nonF_accel, nonF_gyro = normalized_F_accel, normalized_F_gyro, normalized_nonF_accel, normalized_nonF_gyro        

    ### accel
    F_Mean_accelX, F_Mean_accelY, F_Mean_accelZ = [], [], []
    F_std_accelX, F_std_accelY, F_std_accelZ = [], [], []
    F_CV_accelX, F_CV_accelY, F_CV_accelZ = [], [], []
    nonF_Mean_accelX, nonF_Mean_accelY, nonF_Mean_accelZ = [], [], []   
    nonF_CV_accelX, nonF_CV_accelY, nonF_CV_accelZ = [], [], []  
    nonF_std_accelX, nonF_std_accelY, nonF_std_accelZ = [], [], []     
    ### gyro
    F_Mean_gyroX, F_Mean_gyroY, F_Mean_gyroZ = [], [], []
    F_std_gyroX, F_std_gyroY, F_std_gyroZ = [], [], []
    F_CV_gyroX, F_CV_gyroY, F_CV_gyroZ = [], [], []
    nonF_Mean_gyroX, nonF_Mean_gyroY, nonF_Mean_gyroZ = [], [], []   
    nonF_CV_gyroX, nonF_CV_gyroY, nonF_CV_gyroZ = [], [], []  
    nonF_std_gyroX, nonF_std_gyroY, nonF_std_gyroZ = [], [], [] 
 
    for i in range(len(F_accel)):
        ### accel
        F_Mean_accelX.append(np.mean(F_accel[i,:,0]))
        F_Mean_accelY.append(np.mean(F_accel[i,:,1]))
        F_Mean_accelZ.append(np.mean(F_accel[i,:,2]))        
        F_std_accelX.append(np.std(F_accel[i,:,0]))
        F_std_accelY.append(np.std(F_accel[i,:,1]))
        F_std_accelZ.append(np.std(F_accel[i,:,2]))        
        F_CV_accelX.append(np.std(F_accel[i,:,0])/np.mean(F_accel[i,:,0]))
        F_CV_accelY.append(np.std(F_accel[i,:,1])/np.mean(F_accel[i,:,1]))
        F_CV_accelZ.append(np.std(F_accel[i,:,2])/np.mean(F_accel[i,:,2]))        
        ### gyro
        F_Mean_gyroX.append(np.mean(F_gyro[i,:,0]))
        F_Mean_gyroY.append(np.mean(F_gyro[i,:,1]))
        F_Mean_gyroZ.append(np.mean(F_gyro[i,:,2]))        
        F_std_gyroX.append(np.std(F_gyro[i,:,0]))
        F_std_gyroY.append(np.std(F_gyro[i,:,1]))
        F_std_gyroZ.append(np.std(F_gyro[i,:,2]))       
        F_CV_gyroX.append(np.std(F_gyro[i,:,0])/np.mean(F_gyro[i,:,0]))
        F_CV_gyroY.append(np.std(F_gyro[i,:,1])/np.mean(F_gyro[i,:,1]))
        F_CV_gyroZ.append(np.std(F_gyro[i,:,2])/np.mean(F_gyro[i,:,2]))

    ### accel    
    F_Mean_accel = np.swapaxes(np.asarray([F_Mean_accelX,F_Mean_accelY,F_Mean_accelZ]),0,1)
    F_std_accel = np.swapaxes(np.asarray([F_std_accelX, F_std_accelY, F_std_accelZ]),0,1)
    F_CV_accel = np.swapaxes(np.asarray([F_CV_accelX,F_CV_accelY,F_CV_accelZ]),0,1)    
    ### gyro
    F_Mean_gyro = np.swapaxes(np.asarray([F_Mean_gyroX,F_Mean_gyroY,F_Mean_gyroZ]),0,1)
    F_std_gyro = np.swapaxes(np.asarray([F_std_gyroX, F_std_gyroY, F_std_gyroZ]),0,1)
    F_CV_gyro = np.swapaxes(np.asarray([F_CV_gyroX,F_CV_gyroY,F_CV_gyroZ]),0,1)

    for i in range(len(nonF_accel)):            
        ### accel
        nonF_Mean_accelX.append(np.mean(nonF_accel[i,:,0]))
        nonF_Mean_accelY.append(np.mean(nonF_accel[i,:,1]))
        nonF_Mean_accelZ.append(np.mean(nonF_accel[i,:,2]))        
        nonF_std_accelX.append(np.std(nonF_accel[i,:,0]))
        nonF_std_accelY.append(np.std(nonF_accel[i,:,1]))
        nonF_std_accelZ.append(np.std(nonF_accel[i,:,2]))        
        nonF_CV_accelX.append(np.std(nonF_accel[i,:,0])/np.mean(nonF_accel[i,:,0]))
        nonF_CV_accelY.append(np.std(nonF_accel[i,:,1])/np.mean(nonF_accel[i,:,1]))
        nonF_CV_accelZ.append(np.std(nonF_accel[i,:,2])/np.mean(nonF_accel[i,:,2]))       
        ### gyro
        nonF_Mean_gyroX.append(np.mean(nonF_gyro[i,:,0]))
        nonF_Mean_gyroY.append(np.mean(nonF_gyro[i,:,1]))
        nonF_Mean_gyroZ.append(np.mean(nonF_gyro[i,:,2]))        
        nonF_std_gyroX.append(np.std(nonF_gyro[i,:,0]))
        nonF_std_gyroY.append(np.std(nonF_gyro[i,:,1]))
        nonF_std_gyroZ.append(np.std(nonF_gyro[i,:,2]))        
        nonF_CV_gyroX.append(np.std(nonF_gyro[i,:,0])/np.mean(nonF_gyro[i,:,0]))
        nonF_CV_gyroY.append(np.std(nonF_gyro[i,:,1])/np.mean(nonF_gyro[i,:,1]))
        nonF_CV_gyroZ.append(np.std(nonF_gyro[i,:,2])/np.mean(nonF_gyro[i,:,2]))             
       
    #### accel                
    nonF_Mean_accel = np.swapaxes(np.asarray([nonF_Mean_accelX,nonF_Mean_accelY,nonF_Mean_accelZ]),0,1)
    nonF_std_accel = np.swapaxes(np.asarray([nonF_std_accelX, nonF_std_accelY, nonF_std_accelZ]),0,1)
    nonF_CV_accel = np.swapaxes(np.asarray([nonF_CV_accelX,nonF_CV_accelY,nonF_CV_accelZ]),0,1)
    ### statistics features of normalized accel signals
    F_accel_statistics = np.concatenate((F_Mean_accel, F_CV_accel, F_std_accel),axis=1)
    nonF_accel_statistics = np.concatenate((nonF_Mean_accel, nonF_CV_accel, nonF_std_accel),axis=1)    
    
    #### gyro
    nonF_Mean_gyro = np.swapaxes(np.asarray([nonF_Mean_gyroX,nonF_Mean_gyroY,nonF_Mean_gyroZ]),0,1)
    nonF_std_gyro = np.swapaxes(np.asarray([nonF_std_gyroX, nonF_std_gyroY, nonF_std_gyroZ]),0,1)
    nonF_CV_gyro = np.swapaxes(np.asarray([nonF_CV_gyroX,nonF_CV_gyroY,nonF_CV_gyroZ]),0,1)
    ### statistics features of normalized gyro signals
    F_gyro_statistics = np.concatenate((F_Mean_gyro, F_CV_gyro, F_std_gyro),axis=1)
    nonF_gyro_statistics = np.concatenate((nonF_Mean_gyro, nonF_CV_gyro, nonF_std_gyro),axis=1)      
    
    return F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics

##########################################################################################################
def Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_Y, nonF_Y):

    ### Create test set
    F_patient_label_test, nonF_patient_label_test = [], []
    F_seg_no_test, nonF_seg_no_test = [], []
    F_bio_test, nonF_bio_test = [], []
    F_Y_test, nonF_Y_test = [], []
    
    F_accel_statistics_test, nonF_accel_statistics_test = [], []
    F_gyro_statistics_test, nonF_gyro_statistics_test = [], []
        
    for j in test_F_patients:
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_patient_label_test.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_test.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_statistics_test.append(F_accel_statistics[i:i+1,:])
                F_gyro_statistics_test.append(F_gyro_statistics[i:i+1,:])
                F_bio_test.append(F_bio[i:i+1,:])
                F_Y_test.append(F_Y[i:i+1,:])
                
    for j in test_nonF_patients:
        for i in range(len(nonF_subject_no)):        
            if (nonF_subject_no[i]==j):
                nonF_patient_label_test.append(np.asarray(nonF_subject_no[i]).reshape(1,1)) 
                nonF_seg_no_test.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_statistics_test.append(nonF_accel_statistics[i:i+1,:])
                nonF_gyro_statistics_test.append(nonF_gyro_statistics[i:i+1,:])
                nonF_bio_test.append(nonF_bio[i:i+1,:])
                nonF_Y_test.append(nonF_Y[i:i+1,:])
 
    F_patient_label_test = np.concatenate(F_patient_label_test, axis=0)
    F_seg_no_test = np.concatenate(F_seg_no_test, axis=0)     
    F_accel_statistics_test = np.concatenate(F_accel_statistics_test, axis=0)
    F_gyro_statistics_test = np.concatenate(F_gyro_statistics_test, axis=0)    
    F_X_test = np.concatenate((F_accel_statistics_test, F_gyro_statistics_test), axis=1)

    F_bio_test = np.concatenate(F_bio_test, axis=0)
    F_Y_test = np.concatenate(F_Y_test, axis=0)

    nonF_patient_label_test = np.concatenate(nonF_patient_label_test, axis=0)  
    nonF_seg_no_test = np.concatenate(nonF_seg_no_test, axis=0)  
    nonF_accel_statistics_test = np.concatenate(nonF_accel_statistics_test, axis=0)
    nonF_gyro_statistics_test = np.concatenate(nonF_gyro_statistics_test, axis=0)    
    nonF_X_test = np.concatenate((nonF_accel_statistics_test, nonF_gyro_statistics_test), axis=1)
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
def training(test_data, training_F_patients, training_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_Y, nonF_Y):
    [patient_label_test, seg_no_test, X_test, F_test_number, nonF_test_number, bio_test, Y_test] = test_data
               
    train_score, test_score= [], []
    train_sensitivity, train_specificity, train_AUC = [], [], []
    test_sensitivity, test_specificity, test_AUC = [], [], []     
    
    ### Separate training patients
    F_Y_train, F_patient_label_train, F_seg_no_train = [], [], []
    nonF_Y_train, nonF_patient_label_train, nonF_seg_no_train = [], [], [] 
    F_bio_train, nonF_bio_train = [], []
    
    F_accel_statistics_train, nonF_accel_statistics_train = [], []
    F_gyro_statistics_train, nonF_gyro_statistics_train = [], []

    for j in training_F_patients: 
        for i in range(len(F_subject_no)):
            if (F_subject_no[i]==j):
                F_Y_train.append(F_Y[i])
                F_patient_label_train.append(np.asarray(F_subject_no[i]).reshape(1,1))
                F_seg_no_train.append(np.asarray(F_seg_no[i]).reshape(1,1))
                F_accel_statistics_train.append(np.asarray(F_accel_statistics[i:i+1,:]))
                F_gyro_statistics_train.append(np.asarray(F_gyro_statistics[i:i+1,:]))
                F_bio_train.append(np.asarray(F_bio[i:i+1,:]))
          
    for j in training_nonF_patients:
        for i in range(len(nonF_subject_no)):    
            if (nonF_subject_no[i]==j):    
                nonF_Y_train.append(nonF_Y[i])
                nonF_patient_label_train.append(np.asarray(nonF_subject_no[i]).reshape(1,1))
                nonF_seg_no_train.append(np.asarray(nonF_seg_no[i]).reshape(1,1))
                nonF_accel_statistics_train.append(np.asarray(nonF_accel_statistics[i:i+1,:]))
                nonF_gyro_statistics_train.append(np.asarray(nonF_gyro_statistics[i:i+1,:]))
                nonF_bio_train.append(np.asarray(nonF_bio[i:i+1,:]))

    F_Y_train = np.concatenate(F_Y_train, axis=0)
    F_patient_label_train = np.concatenate(F_patient_label_train, axis=0)
    F_seg_no_train = np.concatenate(F_seg_no_train, axis=0)
    F_accel_statistics_train = np.concatenate(F_accel_statistics_train, axis=0)
    F_gyro_statistics_train = np.concatenate(F_gyro_statistics_train, axis=0)
    F_X_train = np.concatenate((F_accel_statistics_train, F_gyro_statistics_train), axis=1)
    F_bio_train = np.concatenate(F_bio_train, axis=0)

    nonF_Y_train = np.concatenate(nonF_Y_train, axis=0)
    nonF_patient_label_train = np.concatenate(nonF_patient_label_train, axis=0)
    nonF_seg_no_train = np.concatenate(nonF_seg_no_train, axis=0)
    nonF_accel_statistics_train = np.concatenate(nonF_accel_statistics_train, axis=0)
    nonF_gyro_statistics_train = np.concatenate(nonF_gyro_statistics_train, axis=0)
    nonF_X_train = np.concatenate((nonF_accel_statistics_train, nonF_gyro_statistics_train), axis=1)
    nonF_bio_train = np.concatenate(nonF_bio_train, axis=0)

    ######################################   
    Y_train, patient_label_train, seg_no_train = [], [], []
    X_train, bio_train = [], []
    
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
    ##########################################################################################      

    ### Train and Predict SVM 
    model = SVC(kernel='linear')
    ##############
    X_train_SVM = X_train[:,:9]  ### Accel statistics only
    # X_train_SVM = X_train[:,9:]  ### Gyro statistics only    
    # X_train_SVM = np.concatenate((X_train[:,9:], bio_train), axis=1) ### Bio + Gyro statistics only 
   
    ##############
    X_test_SVM = X_test[:,:9]  ### Accel statistics only
    # X_test_SVM = X_test[:,9:]  ### Gyro statistics only
    # X_test_SVM = np.concatenate((X_test[:,9:], bio_test), axis=1) ### Bio + Gyro statistics only

    ##############
    X_train = X_train_SVM
    X_test = X_test_SVM   
     
    ##############  
    model.fit(X_train, Y_train)

    train_probabilities = model.predict(X_train)
    train_probabilities = train_probabilities.reshape(train_probabilities.shape[0],1)
    Y_pred_train = np.insert(train_probabilities, 0, 0.5, axis =1)
    Y_pred_train = np.argmax(Y_pred_train, axis=1)
    Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0], 1)

    test_probabilities = model.predict(X_test)
    test_probabilities = test_probabilities.reshape(test_probabilities.shape[0],1)

    Y_pred_test = np.insert(test_probabilities, 0, 0.5, axis =1)
    Y_pred_test = np.argmax(Y_pred_test, axis=1)
    Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0], 1)
    
    train_scores = model.score(X_train, Y_train)       
    test_scores = model.score(X_test, Y_test) 
    train_score = train_scores*100
    test_score = test_scores*100

    ###################################################
    ###################################################
    train_cnf_matrix = confusion_matrix(Y_train, Y_pred_train) # A matrix of truth on row and prediction on column
    train_sensitivity, train_specificity, train_tp, train_fp, train_tn, train_fn, train_accuracy = Sensitivity_And_specificity(train_cnf_matrix) 
 
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
       
    test_segment_results = [test_probabilities, test_cnf_matrix, test_score, test_ROC, test_AUC, test_sensitivity, test_specificity]
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
        Y_Test.append(Y_Test_subject[0])
 
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
    data = [X_test, X_train, Y_train, patient_label_train]    
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
    print("Bootstrp iteration:", i+1)
    print('\n')
    K = 100  ### Multiplier of the length of resampling of each train, val, and test set
    
    ### Train, val and test Faller subjects
    train_F_patients, test_F_patients = train_test_split(np.unique(F_subject_no), test_size=0.2)   ### SUBJECT-LEVEL: np.unique to make sure that the train and test set are separated at subject level    
    training_F_patients = train_F_patients[:int(0.75*len(train_F_patients))]
    val_F_patients = train_F_patients[int(0.75*len(train_F_patients)):]   
    
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
   
    #########################################################################################################
    ##### Get statistics of the signals 
   
    F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics = signal_statistics(normalized_F_accel, normalized_nonF_accel, normalized_F_gyro, normalized_nonF_gyro)
    accel_statistics = np.concatenate((F_accel_statistics, nonF_accel_statistics), axis=0)
    gyro_statistics = np.concatenate((F_gyro_statistics, nonF_gyro_statistics), axis=0)
    
    test_data = Test_set(test_F_patients, test_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_Y, nonF_Y)    
    [patient_label_test, seg_no_test, X_test, F_test_number, nonF_test_number, bio_test, Y_test] = test_data
    F_test_numbers.append(F_test_number)
    nonF_test_numbers.append(nonF_test_number)   
      
    model, data, Test_Overal_results, test_segment_results, train_segment_results, test_seg_thresholds, Test_overall_thresholds = training(test_data, training_F_patients, training_nonF_patients, F_bio, nonF_bio, F_seg_no, nonF_seg_no, F_accel_statistics, nonF_accel_statistics, F_gyro_statistics, nonF_gyro_statistics, F_Y, nonF_Y)
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
###################################################################

#### Print the Computation Time
t1 = time.time()
print("running time (hours):", (t1-t0)/3600)





