# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:24:08 2021

@author: Venus
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# Load the results from the CSV file
pred_results = pd.read_excel("C:\\Users\\Venus\\Documents\\Dissertation\\melspect-model-comparison.xlsx")

# ----- Se+Sp-1 -------
### CNN
df = pred_results[['N_CNN_Gyro_SESP']]
N_CNN_Gyro_SESP = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_SESP']]
R_CNN_Gyro_SESP = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_SESP']]
L_CNN_Gyro_SESP = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_SESP']]
N_CNN_Accel_SESP = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_SESP']]
R_CNN_Accel_SESP = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_SESP']]
L_CNN_Accel_SESP = pd.DataFrame.to_numpy(df)

# ----- Se -------
### CNN
df = pred_results[['N_CNN_Gyro_SE']]
N_CNN_Gyro_SE = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_SE']]
R_CNN_Gyro_SE = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_SE']]
L_CNN_Gyro_SE = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_SE']]
N_CNN_Accel_SE = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_SE']]
R_CNN_Accel_SE = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_SE']]
L_CNN_Accel_SE = pd.DataFrame.to_numpy(df)


# ----- Sp -------
### CNN
df = pred_results[['N_CNN_Gyro_SP']]
N_CNN_Gyro_SP = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_SP']]
R_CNN_Gyro_SP = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_SP']]
L_CNN_Gyro_SP = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_SP']]
N_CNN_Accel_SP = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_SP']]
R_CNN_Accel_SP = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_SP']]
L_CNN_Accel_SP = pd.DataFrame.to_numpy(df)


# ----- Extract F1-scores -------
### CNN
df = pred_results[['N_CNN_Gyro_F1']]
N_CNN_Gyro_F1 = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_F1']]
R_CNN_Gyro_F1 = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_F1']]
L_CNN_Gyro_F1 = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_F1']]
N_CNN_Accel_F1 = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_F1']]
R_CNN_Accel_F1 = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_F1']]
L_CNN_Accel_F1 = pd.DataFrame.to_numpy(df)


# Extract PPV
### CNN
df = pred_results[['N_CNN_Gyro_PPV']]
N_CNN_Gyro_PPV = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_PPV']]
R_CNN_Gyro_PPV = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_PPV']]
L_CNN_Gyro_PPV = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_PPV']]
N_CNN_Accel_PPV = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_PPV']]
R_CNN_Accel_PPV = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_PPV']]
L_CNN_Accel_PPV = pd.DataFrame.to_numpy(df)


# Extract NPV
###CNN
df = pred_results[['N_CNN_Gyro_NPV']]
N_CNN_Gyro_NPV = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Gyro_NPV']]
R_CNN_Gyro_NPV = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Gyro_NPV']]
L_CNN_Gyro_NPV = pd.DataFrame.to_numpy(df)

df = pred_results[['N_CNN_Accel_NPV']]
N_CNN_Accel_NPV = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_NPV']]
R_CNN_Accel_NPV = pd.DataFrame.to_numpy(df)
df = pred_results[['L_CNN_Accel_NPV']]
L_CNN_Accel_NPV = pd.DataFrame.to_numpy(df)


#Extract AUC
df = pred_results[['N_CNN_Gyro_AUC']]
N_CNN_Gyro_auc = pd.DataFrame.to_numpy(df)

df = pred_results[['R_CNN_Gyro_AUC']]
R_CNN_Gyro_auc = pd.DataFrame.to_numpy(df)
df = pred_results[['N_CNN_Accel_AUC']]
N_CNN_Accel_auc = pd.DataFrame.to_numpy(df)
df = pred_results[['R_CNN_Accel_AUC']]
R_CNN_Accel_auc = pd.DataFrame.to_numpy(df)

#############################
N_CNN_Gyro_F1 = N_CNN_Gyro_F1[:-44]   #### Remove Nan value s from the end of dataframe.
N_CNN_Gyro_PPV = N_CNN_Gyro_PPV[:-44]  
N_CNN_Gyro_NPV = N_CNN_Gyro_NPV[:-44]  
N_CNN_Gyro_auc = N_CNN_Gyro_auc[:-44]  
N_CNN_Gyro_SESP = N_CNN_Gyro_SESP[:-44]
N_CNN_Gyro_SE = N_CNN_Gyro_SE[:-44]
N_CNN_Gyro_SP = N_CNN_Gyro_SP[:-44] 

N_CNN_Accel_F1 = N_CNN_Accel_F1[:-37]   #### Remove Nan values from the end of dataframe.
N_CNN_Accel_PPV = N_CNN_Accel_PPV[:-37]  
N_CNN_Accel_NPV = N_CNN_Accel_NPV[:-37]
N_CNN_Accel_auc = N_CNN_Accel_auc[:-37]
N_CNN_Accel_SESP = N_CNN_Accel_SESP[:-37]
N_CNN_Accel_SE = N_CNN_Accel_SE[:-37]
N_CNN_Accel_SP = N_CNN_Accel_SP[:-37]

R_CNN_Gyro_F1 = R_CNN_Gyro_F1[:-13]   #### Remove Nan values from the end of dataframe.
R_CNN_Gyro_PPV = R_CNN_Gyro_PPV[:-13]  
R_CNN_Gyro_NPV = R_CNN_Gyro_NPV[:-13]
R_CNN_Gyro_auc = R_CNN_Gyro_auc[:-13]
R_CNN_Gyro_SESP = R_CNN_Gyro_SESP[:-13]

R_CNN_Accel_F1 = R_CNN_Accel_F1[:-19]   #### Remove Nan values from the end of dataframe.
R_CNN_Accel_PPV = R_CNN_Accel_PPV[:-19]  
R_CNN_Accel_NPV = R_CNN_Accel_NPV[:-19]
R_CNN_Accel_auc = R_CNN_Accel_auc[:-19]
R_CNN_Accel_SESP = R_CNN_Accel_SESP[:-19]

L_CNN_Gyro_F1 = L_CNN_Gyro_F1[:-7]   #### Remove Nan values from the end of dataframe.
L_CNN_Gyro_PPV = L_CNN_Gyro_PPV[:-7]  
L_CNN_Gyro_NPV = L_CNN_Gyro_NPV[:-7]
L_CNN_Gyro_SESP = L_CNN_Gyro_SESP[:-7]

L_CNN_Accel_F1 = L_CNN_Accel_F1[:-7]   #### Remove Nan values from the end of dataframe.
L_CNN_Accel_PPV = L_CNN_Accel_PPV[:-7]  
L_CNN_Accel_NPV = L_CNN_Accel_NPV[:-7]
L_CNN_Accel_SESP = L_CNN_Accel_SESP[:-7]

###############################################################################
SESP_dict = {'$Neck_{CNN_{w}}$': N_CNN_Gyro_SESP, '$Right_{CNN_{w}}$': R_CNN_Gyro_SESP, '$Left_{CNN_{w}}$': L_CNN_Gyro_SESP,
           '$Neck_{CNN_{a}}$': N_CNN_Accel_SESP, '$Right_{CNN_{a}}$': R_CNN_Accel_SESP, '$Left_{CNN_{a}}$': L_CNN_Accel_SESP}

SE_dict = {'$Neck_{CNN_{w}}$': N_CNN_Gyro_SE,
           '$Neck_{CNN_{a}}$': N_CNN_Accel_SE}

SP_dict = {'$Neck_{CNN_{w}}$': N_CNN_Gyro_SP,
           '$Neck_{CNN_{a}}$': N_CNN_Accel_SP}

F1_dict = {'$Neck_{CNN_{w}}$': N_CNN_Gyro_F1, '$Right_{CNN_{w}}$': R_CNN_Gyro_F1, '$Left_{CNN_{w}}$': L_CNN_Gyro_F1,
           '$Neck_{CNN_{a}}$': N_CNN_Accel_F1, '$Right_{CNN_{a}}$': R_CNN_Accel_F1, '$Left_{CNN_{a}}$': L_CNN_Accel_F1}

auc_dict = {'$Neck_{CNN_{w}}$': N_CNN_Gyro_auc, '$Right_{CNN_{w}}$': R_CNN_Gyro_auc,
            '$Neck_{CNN_{a}}$': N_CNN_Accel_auc, '$Right_{CNN_{a}}$': R_CNN_Accel_auc}  # print(acc_dict['$Neck_{CNN_{w}}$'].shape)

SESP_names = ['$Neck_{CNN_{w}}$', '$Neck_{CNN_{a}}$',
              '$Right_{CNN_{w}}$', '$Right_{CNN_{a}}$', 
              '$Left_{CNN_{w}}$', '$Left_{CNN_{a}}$']
F1_names = ['$Neck_{CNN_{w}}$', '$Neck_{CNN_{a}}$']
AUC_names = ['$Neck_{CNN_{w}}$', '$Neck_{CNN_{a}}$']

Names = []
Results = []

# model_names = AUC_names
model_names = SESP_names

# ylabel = 'Sensitivity (%)'
# ylabel = 'Specificity (%)'
# ylabel = 'F1_score'
ylabel = 'AUC'
# ylabel = 'J Index'

# plot_dict = SE_dict
# plot_dict = SP_dict
# plot_dict = F1_dict
plot_dict = auc_dict
# plot_dict = SESP_dict

# settings
matplotlib.rcParams["figure.dpi"] = 550
plt.rcParams["figure.figsize"] = (2,2.5) 


for name in model_names:
    for key in plot_dict.keys():   
        if name ==  key:         
            bootstrap_result = plot_dict[key]
            bootstrap_result = np.squeeze(bootstrap_result)
            # print(key)
            Results.append(bootstrap_result)
            Names.append(name)
#boxplot algorithm comparison
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(Results)
ax.set_xticklabels(Names, fontsize=5.8)
plt.yticks(fontsize=5.7)
plt.ylabel(ylabel, fontsize=7)
plt.show()   
    
    