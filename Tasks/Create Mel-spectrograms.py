#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:09:40 2020

@author: venusroshdi
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib
matplotlib.rcParams["figure.dpi"] = 350 
filepath='C:\\Users\\Venus\\Documents\\Dissertation\\Back\\'
accel_path = filepath + 'Accel\\'
gyro_path = filepath +'Gyro\\'
  
feature = 'gyro'  # 'accel' or 'gyro': to extract the spectrograms from its signals
###########################################################################
normalized_F_raw_accel = np.load(accel_path+'normalized_F_raw_X.npy')
normalized_nonF_raw_accel = np.load(accel_path+'normalized_nonF_raw_X.npy')
F_raw_accel = np.load(accel_path+'F_raw_X.npy')
nonF_raw_accel = np.load(accel_path+'nonF_raw_X.npy')

F_bag_of_subject_no = np.load(accel_path+'F_bag_of_subject_no.npy')
nonF_bag_of_subject_no = np.load(accel_path+'nonF_bag_of_subject_no.npy')
F_bag_of_seg_no = np.load(accel_path+'F_bag_of_seg_no.npy')
nonF_bag_of_seg_no = np.load(accel_path+'nonF_bag_of_seg_no.npy')

normalized_F_accel = np.load(accel_path + 'normalized_F_segmented_X.npy')
normalized_nonF_accel = np.load(accel_path + 'normalized_nonF_segmented_X.npy')

normalized_F_gyro = np.load(gyro_path + 'normalized_F_segmented_X.npy')
normalized_nonF_gyro = np.load(gyro_path + 'normalized_nonF_segmented_X.npy')

F_patient_num =  np.unique(F_bag_of_subject_no)
nonF_patient_num =  np.unique(nonF_bag_of_subject_no)

F_Y = np.ones((len(normalized_F_accel),1))
nonF_Y = np.zeros((len(normalized_nonF_accel),1))

############################################################
if feature == 'accel':
    X = np.concatenate((normalized_F_accel, normalized_nonF_accel), axis=0)
    saving_path = accel_path
if feature == 'gyro':
    X = np.concatenate((normalized_F_gyro, normalized_nonF_gyro), axis=0)
    saving_path = gyro_path
    
Y = np.concatenate((F_Y, nonF_Y), axis=0)
bag_of_subject_no = np.concatenate((F_bag_of_subject_no, nonF_bag_of_subject_no), axis=0)
print(X.shape, Y.shape) 
print(F_patient_num, nonF_patient_num)

######################################################################################
samplerate = 100
f_max = 50      # 50: Above this value is all black in my data. Default is samplerate/2.
nfft = 150      # 150: length of the FFT window (overlapping after measurement of magnitude versus frequency for the midpoint of time chunk)
hoplength = 5   # 10:  number of samples between successive frames (chunk length)
nmels = 128     # default is 128 

######################################################################################
def signal_to_melspec1(signal, samplerate, f_max, nfft, hoplength, nmels):
    
    # y, sr = librosa.load(librosa.ex('signal'))
    mel_spectrum = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_fft=nfft, hop_length=hoplength, n_mels=nmels, fmax=f_max)
    mel_spectrum_dB = librosa.power_to_db(mel_spectrum, ref=np.max)

    # Reshape mel spectrum, ready to append all spectrums
    reshaped_mel_spectrum = mel_spectrum_dB.reshape(mel_spectrum_dB.shape[0], mel_spectrum_dB.shape[1],1)    
    return mel_spectrum_dB, reshaped_mel_spectrum   

######################################################################################
def signal_to_melspec2(signal, samplerate, f_max, nfft, hoplength, nmels):
    
    D = np.abs(librosa.stft(signal, n_fft=nfft,  hop_length=hoplength))**2
    mel_spectrum = librosa.feature.melspectrogram(S=D, sr=samplerate)
    mel_spectrum_dB = librosa.power_to_db(mel_spectrum, ref=np.max)

    # Reshape mel spectrogram, ready to append all mel spectrogram
    reshaped_mel_spectrum = mel_spectrum_dB.reshape(mel_spectrum_dB.shape[0], mel_spectrum_dB.shape[1],1)   
    return mel_spectrum_dB, reshaped_mel_spectrum 
     
######################################################################################
def signal_to_spec(signal, samplerate, f_max, nfft, hoplength):
    
    # separate the signal to time windows, and apply the Fourier Transform on each time window,
    # But it shows that the signal is concentrated in very small frequency and amplitude ranges.
    D = np.abs(librosa.stft(signal, n_fft=nfft,  hop_length=hoplength))
    
    # adjustment to create spectrogram: transform both the y-axis (frequency) to log scale,
    # and the “color” axis (amplitude) to Decibels, which is kind of the log scale of amplitudes
    spectrogram = librosa.amplitude_to_db(D, ref=np.max)  

    # Reshape spectrogram, ready to append all spectrogram
    reshaped_spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1],1)    
    return spectrogram, reshaped_spectrogram

######################################################################################
def three_channeled_spectrograms(S, samplerate, f_max, nfft, hoplength, nmels):
    three_channeled_MelSpect_1, three_channeled_MelSpect_2, three_channeled_spectrogram = [], [], []
    for axis in [0,1,2]:
        signal = S[:,axis]
        
        # Extraxt Mel spectrogram
        # Display of mel-frequency spectrogram coefficients, with custom arguments for mel filterbank construction (default is fmax=sr/2)
        mel_spectrum_dB, reshaped_mel_spectrum = signal_to_melspec1(signal, samplerate, f_max, nfft, hoplength, nmels)
        three_channeled_MelSpect.append(reshaped_mel_spectrum)
       
        # Extraxt spectrogram
        spectrogram, reshaped_spectrogram = signal_to_spec(signal, samplerate, f_max, nfft, hoplength)
        three_channeled_spectrogram.append(reshaped_spectrogram)
        
    three_channeled_MelSpect_1 = np.concatenate(three_channeled_MelSpect_1, axis=-1)
    three_channeled_MelSpect_2 = np.concatenate(three_channeled_MelSpect_2, axis=-1)
    three_channeled_spectrogram = np.concatenate(three_channeled_spectrogram, axis=-1)      
    three_channeled_MelSpect_1 = np.expand_dims(three_channeled_MelSpect_1, axis=0)
    three_channeled_MelSpect_2 = np.expand_dims(three_channeled_MelSpect_2, axis=0)
    three_channeled_spectrogram = np.expand_dims(three_channeled_spectrogram, axis=0)
    return three_channeled_MelSpect_1, three_channeled_MelSpect_2, three_channeled_spectrogram

######################################################################################  
def normalize(data):  ### Normalization: Mapping to (0,1)
    normalized_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))    
    accel_x, accel_y, accel_z = data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]
    accel_x = (accel_x-np.amin(accel_x))/(np.amax(accel_x)-np.amin(accel_x))
    accel_y = (accel_y-np.amin(accel_y))/(np.amax(accel_y)-np.amin(accel_y))
    accel_z = (accel_z-np.amin(accel_z))/(np.amax(accel_z)-np.amin(accel_z))
    normalized_data[:,:,:,0], normalized_data[:,:,:,1], normalized_data[:,:,:,2] = accel_x, accel_y, accel_z
    return normalized_data

###################################################################################### 
F_Spect, F_MelSpect = [], []
nonF_Spect, nonF_MelSpect = [], []
patient_num = np.concatenate((F_patient_num, nonF_patient_num))

for i in range(len(X)):    
    three_channeled_MelSpect, three_channeled_spectrogram = three_channeled_spectrograms(X[i], samplerate, f_max, nfft, hoplength, nmels)    
    if Y[i,0]==1:           
        F_MelSpect.append(three_channeled_MelSpect)
        F_Spect.append(three_channeled_spectrogram)

    if Y[i,0]==0:
        nonF_MelSpect.append(three_channeled_MelSpect)
        nonF_Spect.append(three_channeled_spectrogram)

####################################################
F_MelSpect = np.concatenate(F_MelSpect, axis=0)
nonF_MelSpect = np.concatenate(nonF_MelSpect, axis=0)

F_Spect = np.concatenate(F_Spect, axis=0)
nonF_Spect = np.concatenate(nonF_Spect, axis=0)


######################## Normalize mel spectrograms, fallers and non fallers all together############################## 
MelSpect = np.concatenate((F_MelSpect, nonF_MelSpect), axis=0)
normalized_MelSpect = normalize(MelSpect)
normalized_F_MelSpect = normalized_MelSpect[:F_MelSpect.shape[0],:,:]
normalized_nonF_MelSpect = normalized_MelSpect[F_MelSpect.shape[0]:,:,:]

######################## Normalize spectrograms, fallers and non fallers all together############################## 
Spect = np.concatenate((F_Spect, nonF_Spect), axis=0)
normalized_Spect = normalize(Spect)
normalized_F_Spect = normalized_Spect[:F_Spect.shape[0],:,:]
normalized_nonF_Spect = normalized_Spect[F_Spect.shape[0]:,:,:]

################################## Save spectrograms ######################################
np.save(saving_path + 'F_MelSpect.npy', F_MelSpect)
np.save(saving_path + 'nonF_MelSpect.npy', nonF_MelSpect)
np.save(saving_path + 'normalized_F_MelSpect.npy', normalized_F_MelSpect)
np.save(saving_path + 'normalized_nonF_MelSpect.npy', normalized_nonF_MelSpect)

np.save(saving_path + 'F_Spect.npy', F_Spect)
np.save(saving_path + 'nonF_Spect.npy', nonF_Spect)
np.save(saving_path + 'normalized_F_Spect.npy', normalized_F_Spect)
np.save(saving_path + 'normalized_nonF_Spect.npy', normalized_nonF_Spect)

# Plot Mel spectrograms################################### 
x, y, z = 0, 1, 2
librosa.display.specshow(nonF_MelSpect[10,:,:,y], sr=samplerate, hop_length=hoplength, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Plot spectrograms
librosa.display.specshow(nonF_Spect[0,:,:,y], sr=samplerate, x_axis='time', y_axis='log')  
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Calculate statistics of spectrograms
F_Spect_X, F_Spect_Y, F_Spect_Z = F_Spect[:,:,:,0], F_Spect[:,:,:,1], F_Spect[:,:,:,2]
nonF_Spect_X, nonF_Spect_Y, nonF_Spect_Z = nonF_Spect[:,:,:,0], nonF_Spect[:,:,:,1], nonF_Spect[:,:,:,2]

F_MelSpect_X, F_MelSpect_Y, F_MelSpect_Z = F_MelSpect[:,:,:,0], F_MelSpect[:,:,:,1], F_MelSpect[:,:,:,2]
nonF_MelSpect_X, nonF_MelSpect_Y, nonF_MelSpect_Z = nonF_MelSpect[:,:,:,0], nonF_MelSpect[:,:,:,1], nonF_MelSpect[:,:,:,2]

# Mean of Mel Spectrograms
print("-------------  Mean of Mel Spectrograms --------------")
print("F_Spect_X: ", np.mean(F_Spect_X))
print("nonF_Spect_X: ", np.mean(nonF_Spect_X))

print("F_MelSpect_X: ", np.mean(F_MelSpect_X))
print("nonF_MelSpect_X: ", np.mean(nonF_MelSpect_X))
print("------------------------------------------------")
print("F_MelSpect_Y: ", np.mean(F_MelSpect_Y))
print("nonF_MelSpect_Y: ", np.mean(nonF_MelSpect_Y))
print("------------------------------------------------")
print("F_MelSpect_Z: ", np.mean(F_MelSpect_Z))
print("nonF_MelSpect_Z: ", np.mean(nonF_MelSpect_Z))
print("------------------------------------------------")

# Mean of Mel Spectrograms over intervals of mel scale bins
# 0:40
print("F_MelSpect_X[:,:40,:]: ", np.mean(F_MelSpect_X[:,:40,:]))
print("nonF_MelSpect_X[:,:40,:]: ", np.mean(nonF_MelSpect_X[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,:40,:]: ", np.mean(F_MelSpect_Y[:,:40,:]))
print("nonF_MelSpect_Y[:,:40,:]: ", np.mean(nonF_MelSpect_Y[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,:40,:]: ", np.mean(F_MelSpect_Z[:,:40,:]))
print("nonF_MelSpect_Z[:,:40,:]: ", np.mean(nonF_MelSpect_Z[:,:40,:]))

# 40:80
print("F_MelSpect_X[:,40:80,:]: ", np.mean(F_MelSpect_X[:,40:80,:]))
print("nonF_MelSpect_X[:,40:80,:]: ", np.mean(nonF_MelSpect_X[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,40:80,:]: ", np.mean(F_MelSpect_Y[:,40:80,:]))
print("nonF_MelSpect_Y[:,40:80,:]: ", np.mean(nonF_MelSpect_Y[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,40:80,:]: ", np.mean(F_MelSpect_Z[:,40:80,:]))
print("nonF_MelSpect_Z[:,40:80,:]: ", np.mean(nonF_MelSpect_Z[:,40:80,:]))

# 80:
print("F_MelSpect_X[:,80:,:]: ", np.mean(F_MelSpect_X[:,80:,:]))
print("nonF_MelSpect_X[:,80:,:]: ", np.mean(nonF_MelSpect_X[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,80:,:]: ", np.mean(F_MelSpect_Y[:,80:,:]))
print("nonF_MelSpect_Y[:,80:,:]: ", np.mean(nonF_MelSpect_Y[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,80:,:]: ", np.mean(F_MelSpect_Z[:,80:,:]))
print("nonF_MelSpect_Z[:,80:,:]: ", np.mean(nonF_MelSpect_Z[:,80:,:]))

        
# Std Dev of Mel Spectrograms
print("-------------  Std Dev of Mel Spectrograms --------------")
print("F_Spect_X: ", np.std(F_Spect_X))
print("nonF_Spect_X: ", np.std(nonF_Spect_X))

print("F_MelSpect_X: ", np.std(F_MelSpect_X))
print("nonF_MelSpect_X: ", np.std(nonF_MelSpect_X))
print("------------------------------------------------")
print("F_MelSpect_Y: ", np.std(F_MelSpect_Y))
print("nonF_MelSpect_Y: ", np.std(nonF_MelSpect_Y))
print("------------------------------------------------")
print("F_MelSpect_Z: ", np.std(F_MelSpect_Z))
print("nonF_MelSpect_Z: ", np.std(nonF_MelSpect_Z))
print("------------------------------------------------")


###### Std Dev of Mel Spectrograms over intervals of mel scale bins ######
##### 0:40
print("F_MelSpect_X[:,:40,:]: ", np.std(F_MelSpect_X[:,:40,:]))
print("nonF_MelSpect_X[:,:40,:]: ", np.std(nonF_MelSpect_X[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,:40,:]: ", np.std(F_MelSpect_Y[:,:40,:]))
print("nonF_MelSpect_Y[:,:40,:]: ", np.std(nonF_MelSpect_Y[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,:40,:]: ", np.std(F_MelSpect_Z[:,:40,:]))
print("nonF_MelSpect_Z[:,:40,:]: ", np.std(nonF_MelSpect_Z[:,:40,:]))

# 40:80
print("F_MelSpect_X[:,40:80,:]: ", np.std(F_MelSpect_X[:,40:80,:]))
print("nonF_MelSpect_X[:,40:80,:]: ", np.std(nonF_MelSpect_X[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,40:80,:]: ", np.std(F_MelSpect_Y[:,40:80,:]))
print("nonF_MelSpect_Y[:,40:80,:]: ", np.std(nonF_MelSpect_Y[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,40:80,:]: ", np.std(F_MelSpect_Z[:,40:80,:]))
print("nonF_MelSpect_Z[:,40:80,:]: ", np.std(nonF_MelSpect_Z[:,40:80,:]))

# 80:
print("F_MelSpect_X[:,80:,:]: ", np.std(F_MelSpect_X[:,80:,:]))
print("nonF_MelSpect_X[:,80:,:]: ", np.std(nonF_MelSpect_X[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,80:,:]: ", np.std(F_MelSpect_Y[:,80:,:]))
print("nonF_MelSpect_Y[:,80:,:]: ", np.std(nonF_MelSpect_Y[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,80:,:]: ", np.std(F_MelSpect_Z[:,80:,:]))
print("nonF_MelSpect_Z[:,80:,:]: ", np.std(nonF_MelSpect_Z[:,80:,:]))


# Max of Mel Spectrograms
print("-------------  Max of Mel Spectrograms --------------")
print("F_Spect_X: ", np.max(F_Spect_X))
print("nonF_Spect_X: ", np.max(nonF_Spect_X))

print("F_MelSpect_X: ", np.max(F_MelSpect_X))
print("nonF_MelSpect_X: ", np.max(nonF_MelSpect_X))
print("------------------------------------------------")
print("F_MelSpect_Y: ", np.max(F_MelSpect_Y))
print("nonF_MelSpect_Y: ", np.max(nonF_MelSpect_Y))
print("------------------------------------------------")
print("F_MelSpect_Z: ", np.max(F_MelSpect_Z))
print("nonF_MelSpect_Z: ", np.max(nonF_MelSpect_Z))
print("------------------------------------------------")


# Mean of Mel Spectrograms over intervals of mel scale bins
# 0:40
print("F_MelSpect_X[:,:40,:]: ", np.max(F_MelSpect_X[:,:40,:]))
print("nonF_MelSpect_X[:,:40,:]: ", np.max(nonF_MelSpect_X[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,:40,:]: ", np.max(F_MelSpect_Y[:,:40,:]))
print("nonF_MelSpect_Y[:,:40,:]: ", np.max(nonF_MelSpect_Y[:,:40,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,:40,:]: ", np.max(F_MelSpect_Z[:,:40,:]))
print("nonF_MelSpect_Z[:,:40,:]: ", np.max(nonF_MelSpect_Z[:,:40,:]))
print("------------------------------------------------")
# 40:80
print("F_MelSpect_X[:,40:80,:]: ", np.max(F_MelSpect_X[:,40:80,:]))
print("nonF_MelSpect_X[:,40:80,:]: ", np.max(nonF_MelSpect_X[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,40:80,:]: ", np.max(F_MelSpect_Y[:,40:80,:]))
print("nonF_MelSpect_Y[:,40:80,:]: ", np.max(nonF_MelSpect_Y[:,40:80,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,40:80,:]: ", np.max(F_MelSpect_Z[:,40:80,:]))
print("nonF_MelSpect_Z[:,40:80,:]: ", np.max(nonF_MelSpect_Z[:,40:80,:]))
print("------------------------------------------------")
# 80:
print("F_MelSpect_X[:,80:,:]: ", np.max(F_MelSpect_X[:,80:,:]))
print("nonF_MelSpect_X[:,80:,:]: ", np.max(nonF_MelSpect_X[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Y[:,80:,:]: ", np.max(F_MelSpect_Y[:,80:,:]))
print("nonF_MelSpect_Y[:,80:,:]: ", np.max(nonF_MelSpect_Y[:,80:,:]))
print("------------------------------------------------")
print("F_MelSpect_Z[:,80:,:]: ", np.max(F_MelSpect_Z[:,80:,:]))
print("nonF_MelSpect_Z[:,80:,:]: ", np.max(nonF_MelSpect_Z[:,80:,:]))
