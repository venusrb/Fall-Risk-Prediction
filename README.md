# Fall-Risk-Prediction


This is a repository for a fall risk prediction project. I propose a machine learning algorithm that can predict risk of fall with hig sensitivity to what an experienced geriatrician can do. This would eventually enhance the cumbersome and time-consuming process of fall risk screening currently used in clinics to detect older adults at increased risk of fall.
&nbsp;
 
# Overview
This project includes 4 main tasks:

- Data Collection

- Statistical Analysis

- Machine Learning Application
  - Data Preprocessing
  - 1D Convolutional Neural Network
  - Support Vector Machine

- Feature Representation
  - Time-Frequency Representation
   - 2D Convolutional Neural Network
  - Temporal Gait Features
   - Logistic Regression

The implemntation Python code, the results and conclusions of each task are disscussed and they are illustrated with some visualizations.  
 
# Data Colection  
&nbsp;

## Population
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gender-Distribution.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Weight-Height-Distribution.png" width=500>  
&nbsp;

## Data Acquisition
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Acceleration.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Neck%20Original%20Angular%20velocity.png" width="500">

&nbsp;


# [Machine Learning Application](https://www.mdpi.com/1424-8220/21/10/3481)

&nbsp;
 
## Data Preprocessing

&nbsp;

## 1D Convolutional Neural Network
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/CNN-Diagram.png)
 
## Support Vector Machine
.
 
## Classification Results
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Molde%20comparison%20(a).png)

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Se%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20Sp%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20AUC%20comparison.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20-%20F1%20comparison.png" width="500">
 &nbsp;

## Grad CAM Visualization

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Faller%20grad%20cam%20heatmap.JPG)
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Non-faller%20grad%20cam%20heatmap.JPG)

 
# Time-Frequency Representation
 
## 2D Convolutional Neural Network

<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20and%20melspect%20comparison%20-%20AUC.png" width="500">
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/TUG%20signals%20and%20melspect%20comparison%20-%20SE.png" width="500">
&nbsp;

# Gait Feature Engineering
&nbsp;

## Detection of Gait Events

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-detection.PNG)
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/gait-event-illustration.png)
![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-detection%20with%20illustration.PNG)
&nbsp;
 
## Temporal Gait Features
<img src="https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Gait-illustration.png" width="1000">
&nbsp;

## Logistic Regression
...

&nbsp;
 
# Findings From Three Ways of Feature Representation

![My image](https://github.com/venusrb/Fall-Risk-Prediction/blob/main/Figures/Figure%205.6.JPG)
&nbsp;

 
# Requirements
&nbsp;


 
 
# Installation


