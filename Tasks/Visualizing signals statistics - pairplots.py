# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:30:26 2020

@author: venusroshdi
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

######################
matplotlib.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (2,2)
#####################################################################
sensor = 'Back'
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\' + sensor + '\\'

    ### Load the RMS of trimmed signals
RMS_F_gyro = np.load(filepath+sensor+'RMS_F_gyro.npy', allow_pickle=True)
RMS_nonF_gyro = np.load(filepath+sensor+'RMS_nonF_gyro.npy', allow_pickle=True)
RMS_F_accel = np.load(filepath+sensor+'RMS_F_accel.npy', allow_pickle=True)
RMS_nonF_accel = np.load(filepath+sensor+'RMS_nonF_accel.npy', allow_pickle=True)

RMS_bio = np.load(filepath+sensor+'RMS_bio.npy')
RMS_Y = np.load(filepath+sensor+'RMS_Y.npy')
    ### Load statistics of RMS signals
RMS_accel_statistics = np.load(filepath+sensor+'RMS_accel_statistics.npy')
RMS_gyro_statistics= np.load(filepath+sensor+'RMS_gyro_statistics.npy')

        ################
accel_statistics_train = np.load(filepath+sensor+'accel_statistics_train.npy')
accel_statistics_test = np.load(filepath+sensor+'accel_statistics_test.npy')
gyro_statistics_train = np.load(filepath+sensor+'gyro_statistics_train.npy')
gyro_statistics_test = np.load(filepath+sensor+'gyro_statistics_test.npy')
bio_train = np.load(filepath+sensor+'bio_train.npy')
bio_test = np.load(filepath+sensor+'bio_test.npy') 

Y_train = np.load(filepath+sensor+'Y_train.npy')
Y_test = np.load(filepath+sensor+'Y_test.npy')

subject_number_train = np.load(filepath+sensor+'subject_number_train.npy')
subject_number_test = np.load(filepath+sensor+'subject_number_test.npy')

### normalized fallers and non-fallers in one visualization
accel_statistics = np.concatenate((accel_statistics_train, accel_statistics_test), axis=0)
gyro_statistics = np.concatenate((gyro_statistics_train, gyro_statistics_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
BP = np.concatenate((bio_train[:,4],bio_test[:,4]), axis=0)
H = np.concatenate((bio_train[:,3],bio_test[:,3]), axis=0)
W = np.concatenate((bio_train[:,2],bio_test[:,2]), axis=0)
Y = np.squeeze(Y)
print(Y.shape, RMS_accel_statistics.shape, gyro_statistics.shape, H.shape)

###############################################################################################
color = []
for i in range(Y.shape[0]):
    if Y[i]==0:
        color.append("Low")
    if Y[i]==1:
        color.append("High")   
color = np.asarray(color)
   ################
RMS_color = []
for i in range(RMS_Y.shape[0]):
    if RMS_Y[i]==0:
        RMS_color.append("Low")
    if RMS_Y[i]==1:
        RMS_color.append("High")   
RMS_color = np.asarray(RMS_color)

################
def PairPlot(feature, H, Y, variables):  ### feature: 'Accel' or 'Gyro'

    if feature =='Accel':
        X = RMS_accel_statistics
        gait = pd.DataFrame({feature+' AP Mean': X[:,0], feature+' AP mean * H': X[:,0]*H,
                              feature+' AP Std. Dev.': X[:,6], feature+' AP Std. Dev. * H': X[:,6]*H,
                              feature+' AP CV': X[:,3], feature+' AP CV * H': X[:,3]*H,
                              feature+' AP Min': X[:,9], feature+' AP Min * H': X[:,9]*H,
                              feature+' AP Max': X[:,12], feature+' AP Max * H': X[:,12]*H,
                              feature+' ML Mean': X[:,1], feature+' ML mean * H': X[:,1]*H,
                              feature+' ML Std. Dev.': X[:,7], feature+' ML Std. Dev. * H': X[:,7]*H,
                              feature+' ML CV': X[:,4], feature+' ML CV * H': X[:,4]*H,
                              feature+' ML Min': X[:,10], feature+' ML Min H': X[:,10]*H,
                              feature+' ML Max': X[:,13], feature+' ML Max * H': X[:,13]*H,
                              feature+' IS Mean': X[:,2], feature+' IS mean * H': X[:,2]*H,
                              feature+' IS Std. Dev.': X[:,8], feature+' IS Std. Dev. * H': X[:,8]*H,
                              feature+' IS CV': X[:,5], feature+' IS CV * H': X[:,5]*H,
                              feature+' IS Min': X[:,11], feature+' IS Min * H': X[:,11]*H,
                              feature+' IS Max': X[:,14], feature+' IS Max * H': X[:,14]*H,
                              'Risk of Fall': color})
    if feature =='Gyro':
        X = RMS_gyro_statistics
        gait = pd.DataFrame({feature+' Roll Mean': X[:,0], feature+' Roll mean * H': X[:,0]*H,
                              feature+' Roll Std. Dev.': X[:,6], feature+' Roll Std. Dev. * H': X[:,6]*H,
                              feature+' Roll CV': X[:,3], feature+' Roll CV * H': X[:,3]*H,
                              feature+' Roll Min': X[:,9], feature+' Roll Min * H': X[:,9]*H,
                              feature+' Roll Max': X[:,12], feature+' Roll Max * H': X[:,12]*H,
                              feature+' Pitch Mean': X[:,1], feature+' Pitch mean * H': X[:,1]*H,
                              feature+' Pitch Std. Dev.': X[:,7], feature+' Pitch Std. Dev. * H': X[:,7]*H,
                              feature+' Pitch CV': X[:,4], feature+' Pitch CV * H': X[:,4]*H,
                              feature+' Pitch Min': X[:,10], feature+' Pitch Min H': X[:,10]*H,
                              feature+' Pitch Max': X[:,13], feature+' Pitch Max * H': X[:,13]*H,
                              feature+' Yaw Mean': X[:,2], feature+' Yaw mean * H': X[:,2]*H,
                              feature+' Yaw Std. Dev.': X[:,8], feature+' Yaw Std. Dev. * H': X[:,8]*H,
                              feature+' Yaw CV': X[:,5], feature+' Yaw CV * H': X[:,5]*H,
                              feature+' Yaw Min': X[:,11], feature+' Yaw Min * H': X[:,11]*H,
                              feature+' Yaw Max': X[:,14], feature+' Yaw Max * H': X[:,14]*H,
                              'Risk of Fall': color})
     
    sns.pairplot(gait, hue='Risk of Fall', hue_order= ['Low','High'],  vars=variables, plot_kws={'alpha':1})  ### Try diag_kw and grid_kw 


def histogram(feature, H, Y, variable):
    if feature == 'Accel':
        X = RMS_accel_statistics
    if feature =='Gyro':
        X = RMS_gyro_statistics
        
    gait = pd.DataFrame({feature+' Mean': X[:,0], feature+' Mean * H': X[:,0]*H,
                          feature+' Std. Dev.': X[:,1], feature+' Std. Dev. * H': X[:,1]*H,
                          feature+' CV': X[:,2], feature+' CV * H': X[:,2]*H,
                          feature+' Min': X[:,3], feature+' Min * H': X[:,3]*H,
                          feature+' Max': X[:,4], feature+' Max * H': X[:,4]*H,
                          'Risk of Fall': RMS_color})
      
    sns.displot(gait, x=variable, hue='Risk of Fall',  hue_order= ['Low','High'], kind='kde', fill=True, alpha=.45, linewidth=2)  

######################################################################################################################
### Pairplots For The Entire Data Set with choice of Accel or Gyro:

feature = 'Accel'
PairPlot(feature, H, Y, variables=[feature+' AP Mean', feature+' ML Mean', feature+' IS Mean'])# figure.savefig("pairplots of mean", dpi=500)
PairPlot(feature, H, Y, variables=[feature+' AP Min', feature+' ML Min', feature+' IS Min'])
PairPlot(feature, H, Y, variables=[feature+' AP Max', feature+' ML Max', feature+' IS Max'])
PairPlot(feature, H, Y, variables=[feature+' AP CV', feature+' ML CV', feature+' IS CV'])
PairPlot(feature, H, Y, variables=[feature+' AP Std. Dev.', feature+' ML Std. Dev.', feature+' IS Std. Dev.'])
PairPlot(feature, H, Y, variables=[feature+' AP Std. Dev. * H', feature+' ML Std. Dev. * H', feature+' IS Std. Dev. * H'])
#####################
feature = 'Gyro'
PairPlot(feature, H, Y, variables=[feature+' Roll Mean', feature+' Pitch Mean', feature+' Yaw Mean'])
PairPlot(feature, H, Y, variables=[feature+' Roll Min', feature+' Pitch Min', feature+' Yaw Min'])
PairPlot(feature, H, Y, variables=[feature+' Roll Max', feature+' Pitch Max', feature+' Yaw Max'])
PairPlot(feature, H, Y, variables=[feature+' Roll CV', feature+' Pitch CV', feature+' Yaw CV'])
PairPlot(feature, H, Y, variables=[feature+' Roll Std. Dev.', feature+' Pitch Std. Dev.', feature+' Yaw Std. Dev.'])
PairPlot(feature, H, Y, variables=[feature+' Roll Std. Dev. * H', feature+' Pitch Std. Dev. * H', feature+' Yaw Std. Dev. * H'])


feature = 'Accel'
histogram(feature, H, Y, variables=[feature+' Mean', feature+' ML Mean', feature+' IS Mean'])# figure.savefig("pairplots of mean", dpi=500)
histogram(feature, H, Y, variables=[feature+' Min', feature+' ML Min', feature+' IS Min'])
histogram(feature, H, Y, variables=[feature+' Max', feature+' ML Max', feature+' IS Max'])
histogram(feature, H, Y, variables=[feature+' CV', feature+' ML CV', feature+' IS CV'])
histogram(feature, H, Y, variables=[feature+' Std. Dev.', feature+' ML Std. Dev.', feature+' IS Std. Dev.'])
histogram(feature, H, Y, variables=[feature+' Std. Dev. * H', feature+' ML Std. Dev. * H', feature+' IS Std. Dev. * H'])
# #####################
feature = 'Gyro'
histogram(feature, H, Y, variable=feature+' Mean')
histogram(feature, H, Y, variable=feature+' Min')
histogram(feature, H, Y, variables=[feature+' Max', feature+' Max', feature+' Max'])
histogram(feature, H, Y, variables=[feature+' CV', feature+' CV', feature+' CV'])
histogram(feature, H, Y, variables=[feature+' Std. Dev.', feature+' Std. Dev.', feature+' Std. Dev.'])
histogram(feature, H, Y, variables=[feature+' Std. Dev. * H', feature+' Std. Dev. * H', feature+' Std. Dev. * H'])

