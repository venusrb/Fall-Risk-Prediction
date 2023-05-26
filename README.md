# Fall-Risk-Prediction

This is a repository for a fall risk prediction project. I propose a machine learning algorithm that can predict risk of fall with high sensitivity to what an experienced geriatrician can do. This would eventually enhance the cumbersome and time-consuming process of fall risk screening currently used in clinics to detect older adults at increased risk of fall.
&nbsp;
 
# Overview

This project has 4 parts:

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
&nbsp;
 
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Weight-Height-Distribution.png" width=500>  
&nbsp;

## Data Acquisition
&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Acceleration.png" width="500">
&nbsp;


<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Angular%20velocity.png" width="500">

&nbsp;

# Statistical Analysis
&nbsp;
 
## Physiological Features
&nbsp;

## Kinematics statistics
For each three-directional acceleration and three-directional angular velocity signals of neck, right and left feet sensors, mean, standard deviation and coefficient of variation (CV) are calculated. These kinematics statistics are considered as separate samples for [faller/non-faller z-tests](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Z-tests%20Faller-nonFaller%20Comparisons.py). The comprehensive analysis of faller/non-faller group sensitivity is described in [my dissertation document](https://doi.org/10.17077/etd.005886). Among all the kinematics statistics, the CV of the roll angular velocity of the neck sensor is found the most significant attribute in distinguishing fallers and non-fallers (p-value=0). However, as shown in the following pairplots [(visualization code)](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Visualizing%20signals%20statistics%20-%20pairplots.py) of fallers/non-fallers' CV of roll angular velocity distributions, the difference between the mean of two groups is only 1% to 4% difference and fallers and non-fallers are not linearlly separable.

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Back%20Gyro%20absolute%20CV.png" width="600">


# [Machine Learning Application](https://www.mdpi.com/1424-8220/21/10/3481)
Machine Learning algorithms are used as a robust non-linear mathematical model to digest the complex multi-dimensional feature space of kinematics time-series of the TUG test and detect the contributing fall-risk factorsand predict risk of fall.
&nbsp;


## Data Preprocessing
- [**Resampling:**](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/io.py) The raw signals are resampled from their original 250 Hz to 100 Hz.
- [**Normalization:**](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/main.py) The Kinematics signals are normalized from their respective range to [0,1] using the minimum and maximum magnitude of acceleration and angular velocity signals across all subjects.
- [**Zero-padding:**](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/io.py) It is used to have all the input signals in the same size. This means that the time length of a subject’s signals is increased by adding zeros to the end of the signals until all the subjects have the same length as the longest TUG test’s length.
- [**Segmentation:**](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/io.py) Motion signals are cut into three-second segments using a sliding window approach. A three-second window slides over a signal with a one-second stride and creates the three-second segments until the sliding window covers the entire signal. Signal segmentation is used to enhance the performance of ML models, specifically the proposed CNN algorithm.
- [**Concatenation:**](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/io.py) Every individual participant’s segmented signals are stacked channel-wise (3 acceleration and 3 angular velocity channels). The three-channel signal segments of each sensor location are considered as the input to the prediction models.

&nbsp;

## Bootstrap Resampling and Bagging
To boost the sample size (in addition to signal segmentation), 100 iterations of bootstrap resampling is performed to estimate the prediction uncertainty due to the sample bias. In each bootstrapping iteration 98 subjects are randomly divided into Test, Validation, and Training subjects as illustrated in the follong figure. Then bootstraping resampling (with replacement) is performed for each sets of subjects for _100*number of subjetcs in the set_ to multiply the size of Test, Validation, and Training subjects by 100 and estimate the distribution of population. Then, Test bag of segments, Validation bag of segments, and Trainign bag of segments are built by palcing all the segments of each subject in the associated set.
&nbsp;

&nbsp;


<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Bootstrap%20Resampling.JPG">

&nbsp;

The training bag of segments is used to train the ML model and the validation bag of segments is used to validate and adjust the model constantly while training. The trained model is used to predict the risk of fall for the Test bag of segments. Finally, The vote of the majority segments of a subject determines its fall-risk status.
This process is repeated for 100 iterations and after bagging the resulted 100 models, the mean and 95% Confidence Interval CI) of the prediction metrics are used to evaluate the bagging model prediction.



## 1D Convolutional Neural Network Model with the Segmented Kinematics Signals of the TUG Test

The CNN model includes:
- Kinematics Feature Extraction: 4 building blocks of 1-D convolutional (Conv) layers, each followed by a Batch Normalization (BN) and ReLU activation, which all together extracted the signals’ high-level gait features. Additionally, Max pooling layers were used after the second and the fourth ReLU activations to downsample the similar local information into a concentrated output.
- Fall-Risk-Classification: The feature maps of the last ReLU activation were flattened into a 1D array and then fed into a fully connected (FC) layer with a sigmoid activation function to serve as the predictor of the fall-risk probability. Then, binary classification of fallers versus non-fallers was performed using the threshold probability of 0.5. 
  
&nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/CNN-Diagram.png)
&nbsp;

```
import keras

# Define a function to build the CNN model that receives X as a numpy input of shape (# of segments, 300, 3)
# Note: each 3-second segment of a 100 Hz signal is equivalent to 300 timestampts of the signal

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
    
model = CNN_model
```
&nbsp;


## Support Vector Machine

A SVM model with a linear kernel were trained with the mean, standard deviation, and coefficient of variation of the three directional signals such that in each experiment, nine statistical variables are the inputs rather than the three-channel time series that are fed into the CNN models.

```
from sklearn.svm import SVC

model = SVC(kernel='linear')
```
&nbsp;


 
## Classification Results
The [bagging of 100 bootstrapping CNNs](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Bootstraping%20CNN%20-%20segmented%20signals.py) and the [bagging of 100 bootstrapping SVMs](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Bootstraping%20SVM%20-%20segmented%20signals.py) are performed for each three-channel acceleration and three-channel angular velocity signals of each of three sensors separately and the mean and 95% CI of prediction metrics are evaluated to [compare the overall peformance of the CNN and SVM models](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Models%20Comparison.py).
&nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Molde%20comparison%20(a).png)

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Se%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Sp%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20AUC%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20F1%20comparison.png" width="500">
 &nbsp;

## Grad CAM Visualization
Utilizing Grad Class Activation Map technique, we can localize the gait segments that are associated with high risk of fall. [The implementation code is provided](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Grad%20CAM%20Visualization%20of%20CNN.py). This could eventually help to the discovery of the problematic gait biomentrics that are creating the high risk of future falls and could help the clinicians to find the appropriate treatments to intervene and decrease the risk of future falls.

&nbsp;

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Faller%20grad%20cam%20heatmap.JPG)

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Non-faller%20grad%20cam%20heatmap.JPG)
&nbsp;

 
# Time-Frequency Representation
&nbsp;

## [Mel-spectrograms](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Tasks/Create%20Mel-spectrograms.py)
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
&nbsp;

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
Logistic regression with the extracted temporal features of the right and left feet can predict risk of fall with 70% sensitivity and 70% specificity. Although it cannot reach the high sensitivity as the CNN model can, it balances the sensitivity and specificity. This could help in future diagnosis studies to detect gait impairments.

&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%2C%20melspect%2C%20gait%20features%20Model%20Comparison%20-%20SE.png" width="600">

&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%2C%20melspect%2C%20gait%20features%20Model%20Comparison%20-%20SP.png" width="600">

&nbsp;

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%2C%20melspect%2C%20gait%20features%20Model%20Comparison%20-%20AUC.png" width="600">

 
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
conda install scikit-learn
conda install matplotlib
conda install seaborn
conda install keras
conda install keras-vis
```

