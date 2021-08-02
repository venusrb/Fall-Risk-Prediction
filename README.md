# Fall-Risk-Prediction


This is a repository for a fall risk prediction project. I propose a machine learning algorithm that can predict risk of fall with hig sensitivity to what an experienced geriatrician can do. This would eventually enhance the cumbersome and time-consuming process of fall risk screening currently used in clinics to detect older adults at increased risk of fall.
&nbsp;
 
# Overview

This project includes 4 main tasks:

- Data Collection [(paper)](https://www.mdpi.com/1424-8220/21/10/3481)
  - Population
  - Data Acquisition 

- Statistical Analysis
  - Physiological Features
  - Kinematics statistics 

- Machine Learning Application [(paper)](https://www.mdpi.com/1424-8220/21/10/3481)
  - Data Preprocessing
  - Bootstrap Resampling and Bagging
  - 1D Convolutional Neural Network (CNN)
  - Support Vector Machine (SVM)

- Feature Representation
  - Time-Frequency Representation
   - 2D Convolutional Neural Network
  - Temporal Gait Features
   - Logistic Regression

The results and conclusions of each task are disscussed and illustrated with some visualizations. In addition, the implemntation Python code for each task is provided under [Tasks](https://github.com/venusrb/Fall-Risk-Prediction/tree/main/Tasks) folder.

 
# Data Colection

98 patients, 65 years old and older, a diverse group of geriatric patients, participated in this study. They were evaluated with the Timed-Up-and-Go (TUG) test while three sensors were installed on their bodies (on their right and left shoes, and the collear of their clothing).
&nbsp;

## Population
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gender-Distribution.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Weight-Height-Distribution.png" width=500>  
&nbsp;

## Data Acquisition
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Acceleration.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Angular%20velocity.png" width="500">

&nbsp;

# Statistical Analysis
&nbsp;
 
## Physiological Features
&nbsp;

## Kinematics statistics
For each three-directional acceleration and three-directional angular velocity signals of neck, right and left feet sensors, mean, standard deviation and coefficient of variation (CV) are calculated. These kinematics statistics are considered as separate samples for faller/non-faller z-tests. Among all the kinematics statistics, the CV of the roll angular velocity of the neck sensor is found the most significant attribute in distinguishing fallers and non-fallers (p-value=0). However, as shown in the following pairplots of fallers/non-fallers' CV of roll angular velocity distributions, the difference between the mean of two groups is only 1% to 4% difference and fallers and non-fallers are not linearlly separable.

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Back%20Gyro%20absolute%20CV.png" width="600">

# [Machine Learning Application](https://www.mdpi.com/1424-8220/21/10/3481)

&nbsp;
 
## Data Preprocessing
- Normalization: The Kinematics signals are normalized from their respective range to [0,1] using the minimum and maximum magnitude of acceleration and angular velocity signals across all subjects.
- Zero-padding was used to have all the input signals in the same size. This means that the time length of a subject’s signals is increased by adding zeros to the end of the signals until all the subjects have the same length as the longest TUG test’s length.
- Segmentation: Motion signals were cut into three-second segments using a sliding window approach. A three-second window slides over a signal with a one-second stride and creates the three-second segments until the sliding window covers the entire signal. Signal segmentation was used to enhance the performance of ML models, specifically the proposed CNN algorithm.
- Concatenation: Every individual participant’s segmented signals were stacked channel-wise (3 acceleration and 3 angular velocity channels). The three-channel signal segments of each sensor location were considered as the input to the prediction models.

&nbsp;

## Bootstrap Resampling and Bagging
To boost the sample size (in addition to signal segmentation), 100 iterations of bootstrap resampling is performed to estimate the prediction uncertainty due to the sample bias. In each bootstrapping iteration 98 subjects are randomly divided into Test, Validation, and Training subjects as illustrated in the follong figure. Then bootstraping resampling (with replacement) is performed for each sets of subjects for _{100*number of subjetcs in the set}_ to built the multiply the size of Test, Validation, and Training subjects by 100 and estimate the distribution of population.

&nbsp;
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Bootstrap%20Resampling.JPG">

## 1D Convolutional Neural Network Model with the Segmented Kinematics Signals of the TUG Test

The [CNN]() model includes:
- Kinematics Feature Extraction: 4 building blocks of 1-D convolutional (Conv) layers, each followed by a Batch Normalization (BN) and ReLU activation, which all together extracted the signals’ high-level gait features. Additionally, Max pooling layers were used after the second and the fourth ReLU activations to downsample the similar local information into a concentrated output.
- Fall-Risk-Classification: The feature maps of the last ReLU activation were flattened into a 1D array and then fed into a fully connected (FC) layer with a sigmoid activation function to serve as the predictor of the fall-risk probability. Then, binary classification of fallers versus non-fallers was performed using the threshold probability of 0.5. 
 &nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/CNN-Diagram.png)
&nbsp;

## Support Vector Machine

A [SVM](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Bootstraping%20SVM%20-%20segmented%20signals.py) model with a linear kernel were trained with the mean, standard deviation, and coefficient of variation of the three directional signals such that in each experiment, nine statistical variables were the inputs rather than the three-channeled time series that were fed into the CNN models.

 
## Classification Results
&nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Molde%20comparison%20(a).png)

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Se%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Sp%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20AUC%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20F1%20comparison.png" width="500">
 &nbsp;

## Grad CAM Visualization
&nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Faller%20grad%20cam%20heatmap.JPG)
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Non-faller%20grad%20cam%20heatmap.JPG)
&nbsp;

 
# Time-Frequency Representation
&nbsp;

## Mel-spectrograms
&nbsp;

### Right Foot Pitch Angular Velocity Mel-spectrogram of a Faller's TUG Test 3-second Segment Example

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Right%20gyro%20y%20melspect%20faller0.png" width="500">
&nbsp;

### Right Foot Pitch Angular Velocity Mel-spectrogram of a Non-faller's TUG Test 3-second Segment Example
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Right%20gyro%20y%20melspect%20nonfaller10.png" width="500">
&nbsp;

## 2D Convolutional Neural Network Classification Results
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20and%20melspect%20comparison%20-%20AUC.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20and%20melspect%20comparison%20-%20SE.png" width="500">
&nbsp;

# Gait Feature Engineering
&nbsp;


### Exploring Low-pass Filterings to Select the Best Representation of the Right and Left Feet Pitch Angular velocity Signals
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-detection.PNG)
&nbsp;

### Finding the Timing of the Local Maximum Points of the Right and Left Feet Pitch Angular velocity Magnitude
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/gait-event-illustration.png)
&nbsp;

## Detection of Gait Events - Heel-strike (HS) and Toe-off (TO)
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-detection%20with%20illustration.PNG)
&nbsp;
 
## Temporal Gait Features
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-illustration.png" width="1000">
&nbsp;

## Logistic Regression Results
&nbsp;

 
# Findings From Three Ways of Feature Representation

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Figure%205.6.JPG)
&nbsp;

 

# Requirements
- Python 3.7-3.9
- tensorflow>=2.0.0
- keras
&nbsp;


 
# Installation

The following Python packages should be installed for performing the tasks.

```
conda install tqdm
conda install pandas
conda install scipy
conda install matplotlib
conda install seaborn
conda install keras
conda install keras-vis
```

