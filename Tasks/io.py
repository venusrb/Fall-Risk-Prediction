
"""Functions related to input and output of dataset."""

import os
import pandas as pd
import numpy as np
import math

import pathlib
import pickle
from tqdm import tqdm
from scipy import signal

import json
config = json.load(open( os.path.join(*[os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json']) ))

class IMUdata():
    """ Extract - Transform - Load
    Extracts from csv folder
    Transforms when load is called
    user only interacts with `load` function
    """
    def __init__(self, force_recache=False):
        """Instantiate IMUdata
           Reads data from RunScribe csv data files in `imu_dir` and caches them in the folder that this file is in.
           Reading of CSV happens only the first time (i.e. cache doesn't exist) or the user forces to recache by flagging `force_recache=True`.
        Arguments:
            imu_dir: a string. folder name where IMU data files exist
            force_recache: boolean. if `True`, it forces to read CSV files again and recreates a cache.
        """
        cache_name = 'data.pkl'
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(*[cache_dir, cache_name])
        if force_recache==True or not self._check_cache():
            imu       = self._extract_imu()
            code_book = self._read_codebook()
            merged    = self._merge_imu_codebook(imu, code_book)
            TUG_imu, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features = self._TUG_data(merged)
            data = [TUG_imu, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features]


            pickle.dump( data, open( self.cache_path, "wb+" ) )
        else:
            data = self._load_cache()
            
        self.data = data  
    
    def _check_cache(self):
        return os.path.isfile(self.cache_path)
    
    def _load_cache(self):
        return pickle.load( open(self.cache_path, 'rb') )
    
    def _parse_filenames(self, paths):
        """Parse information from filenames of all IMU data files (.csv) within a folder.
           IMU files are named as "<date_of_measurement>_<measured_at_(hhmmss)>_<subject_id>_<test_no>_<sensor_location>_<sensor_id>.csv"
        Arguments:
            paths: a string. folder name where IMU data files (.csv) to be loaded
        Returns:
            data: a dictionary of lists parsed from filenames. keys include:
                - `dates` contains a list of date of measurement (string, formated three characters of the month name followed by numerical date like `Apr22`).
                - `times` contains a list of time of measurement (string, formatted HHMMSS).
                - `subjects` contains a list of subject identifiers (string, formatted subXX, where XX stands for subject identification number.)
                - `tests` contains a list of test numbers (string, formated TestX, where X stands for test number).
                - `sensor_loc` contains a list of sensor locations (string, either `Back`, `Right`, or `Left`).
        """
        filenames = [os.path.split(path)[1] for path in paths]
        parsed = [filename.split('_') for filename in filenames]
        
        data = {'dates':      [p[0] for p in parsed],
                'times':      [p[1] for p in parsed],
                'subjects':   [p[2] for p in parsed],
                'tests':      [p[3] for p in parsed],
                'sensor_loc': [p[4] for p in parsed]}
        
        return data


    def _read_csv(self, path):
        """Read IMU data from RunScribe csv file.
        Arguments:
            path: a string path to RunScribe csv.
        Returns:
            DF: a pandas dataframe containing acceleration and gyroscope time series.
        """
        with open(path) as csv_file:
           
            DF = pd.read_csv(csv_file, sep=None)
            
        # Get rid of the header
        DF = DF[DF['Message'] == 'motion_hires']    

        # Re-index the rows
        length = len(DF.index) # numb. of rows in csv_dataframe
        DF.index = range(length)
        
        # Rename columns and keep only the ones that matter
        DF.rename(columns={'Value 2': 'accel_x', 'Value 3': 'accel_y', 'Value 4': 'accel_z',
                           'Value 5': 'gyro_x', 'Value 6': 'gyro_y', 'Value 7': 'gyro_z'},
                  inplace=True)
        DF = DF[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
        
        # Convert to G-unit for gravity and deg/sec for rotational velocity
        DF['accel_x'] = DF['accel_x'].astype(float)/2048  # To get G's
        DF['accel_y'] = DF['accel_y'].astype(float)/2048
        DF['accel_z'] = DF['accel_z'].astype(float)/2048     
        DF['gyro_x'] = DF['gyro_x'].astype(float)*16.384/256 # To get Deg/Sec
        DF['gyro_y'] = DF['gyro_y'].astype(float)*16.384/256
        DF['gyro_z'] = DF['gyro_z'].astype(float)*16.384/256
       
        return DF
    
    def _extract_imu(self):
        imu_dir = pathlib.Path(config['IMU_DIR'])
        imu_paths = [str(path) for path in list(imu_dir.glob('*.csv'))]
        
        data = self._parse_filenames(imu_paths)
        dfs = [self._read_csv(path) for path in tqdm(imu_paths)]
        
        data['imu'] = dfs
      
        return data

    def _read_codebook(self):

        codebook = pd.read_excel(config['CODEBOOK_PATH']) #### Changed this after installing new version of pandas
        
        codebook.rename(columns={'Age in Years': 'Age', 'Gender'+'\n'+'male=1'+'\n'+'female=2': 'Gender', 'Height\n(meters)': 'Height',
                                 'Weight\n(kilograms)': 'Weight', 'Blood Pressure\n(systolic)':'Blood Pressure'},
                        inplace=True)
        codebook = codebook[['Subject Number', 'Age', 'Gender', 'Height', 'Weight', 'Blood Pressure', 'PreviousFalls',
                             'Fall', 'Falls reported', 'Number of falls', 'Composite Score']]
        
        return codebook
    
    def _merge_imu_codebook(self, imu, codebook):
        N = len(imu['imu'])  # number of imu data available in the database
        
        # Create an empty data frame with total <len(data)> rows
        merged = []
        
        # Parse the dictionary and fill in the empty merged list.
        for i in range(N):
                    
            # Add zero before the subject number if it's less than 10
            if len(imu['subjects'][i]) == 4:
                imu['subjects'][i] = str('Sub00') + str(imu['subjects'][i][-1])
                
            if len(imu['subjects'][i]) == 5:
                imu['subjects'][i] = str('Sub0') + str(imu['subjects'][i][-2]) + str(imu['subjects'][i][-1])  
                
            current_subject = imu['subjects'][i]
            current_sensor_loc = imu['sensor_loc'][i]
            
            # Check if the subject and sensor items of imu dictionary have already existed in the merged list.
            subject_flag = [x[0] == current_subject for x in merged]
            sensor_flag = [x[1] == current_sensor_loc for x in merged]
            
            # Find the index in the merged list in which both subject and sensor items of imu dictionary match with the ones in the merged.
            idx = np.logical_and(subject_flag, sensor_flag)
            idx = np.where(idx)[0]
            
            # Extract the test number of the current imu item
            test_no = int(imu['tests'][i][-1])-1
            
            # Adding rows to the merged list
            
            # Add the entire row of imu dictionary to the merged list if there was no matching found between the imu and merged
            if len(idx) == 0:
                
                tests = [np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]]), np.asarray([[0]])]
                tests[test_no] = imu['imu'][i]
                new_row = [imu['subjects'][i], imu['sensor_loc'][i]] + tests + [codebook['Fall'][int(imu['subjects'][i][3:])-1]] + [codebook['Subject Number'][int(imu['subjects'][i][3:])-1]] + [codebook['Falls reported'][int(imu['subjects'][i][3:])-1]] + [codebook['Number of falls'][int(imu['subjects'][i][3:])-1]] + [codebook['Age'][int(imu['subjects'][i][3:])-1]] + [codebook['Gender'][int(imu['subjects'][i][3:])-1]] + [codebook['Height'][int(imu['subjects'][i][3:])-1]] + [codebook['Weight'][int(imu['subjects'][i][3:])-1]]+ [codebook['Blood Pressure'][int(imu['subjects'][i][3:])-1]] + [codebook['Composite Score'][int(imu['subjects'][i][3:])-1]] 
                merged.append(new_row)
           
            # Add only the test item of imu to the corresponding subject and sensor in the merged list if there is a match
            else:
                idx = idx[0]
                merged[idx][2 + test_no] = imu['imu'][i]                
              
       
        # Sort merged list according to subject number and sensor_loc           
        merged = sorted(merged, key=lambda x: (x[0], x[1]))
        
        return merged


    def _TUG_data(self, merged):
        
        TUG_imu, labels, subject_no = [], [], []
        report_fall, no_of_falls = [], []
        age, gender, height, weight, blood_pressure = [], [], [], [], []
        composite_score = []

        # TUG test is the fifth test
        test_numb = 5
            
        # Create test list by parsing the data list
        for i in range(len(merged)):
            if (merged[i][11]!=55 and merged[i][11]!=79):   ### Exclude Sub55 and Sub79 because 55 doesn't have right sensor of TUG and 79 doesn't have left sensor of TUG           
                if (i%3 == 0):
                    new_row = []
                
                new_row.append(merged[i][test_numb+1])
            
                if (i%3 == 2):               
                    TUG_imu.append(new_row)
                    labels.append(merged[i][10])
                    subject_no.append(merged[i][11])
                    report_fall.append(merged[i][12])
                    no_of_falls.append(merged[i][13])
                    age.append(merged[i][14])
                    gender.append(merged[i][15])
                    height.append(merged[i][16])
                    weight.append(merged[i][17])
                    blood_pressure.append(merged[i][18])
                    composite_score.append(merged[i][19])
                    identity_features = [age, gender, height, weight, blood_pressure]
                
        return TUG_imu, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features


    def _TUG_oneSensor_data(self, TUG_imu, labels, subject_no, report_fall, no_of_falls, sensors):
        # Find the number of sensors which we want to include
        sen_numbers = []
        for i in range(len(sensors)):
            if sensors[i] == 'back':
                sen_number = 0
            if sensors[i] == 'left':
                sen_number = 1            
            if sensors[i] == 'right':
                sen_number = 2
            sen_numbers.append(sen_number)
        
        sen_numbers.sort()
        TUG_one_Sensor_imu = []
        updated_subject_no = []
        updated_report_fall = []
        updated_no_of_falls = []
        updated_labels = []
        for i in range(len(TUG_imu)):            

            new_row = []
            for j in sen_numbers:

               if TUG_imu[i][j].shape[0]!=1: ## In TUG_imu[i] for jth elemnt/sensor, the value inside is a numpy array of shape (1,1) if there is no TUG for that sensor
                   new_row.append(TUG_imu[i][j])
                   updated_subject_no.append(subject_no[i])
                   updated_report_fall.append(report_fall[i])
                   updated_no_of_falls.append(no_of_falls[i])
                   updated_labels.append(labels[i])

            # Find the min number of rows among the selected sensors in the new_row
            if len(new_row)>=1:
                min_row = new_row[0].shape[0]
                for k in range(len(new_row)):
                    if (new_row[k].shape[0] <= min_row):
                        min_row = new_row[k].shape[0]
            
                new_imu = []
                for k in range(len(new_row)):
                    new_imu.append(new_row[k][:min_row][:])

                TUG_one_Sensor_imu.append(new_imu)
            
        subject_no = updated_subject_no
        report_fall = updated_report_fall
        no_of_falls = updated_no_of_falls
        labels = updated_labels
        return TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls

    def _selected_features_data(self, data, features_list):
        
        features = []
        for timestep in range(len(data)):

            new_features = [data[timestep][feature] for feature in features_list]
            features.append(new_features)
        features = np.asarray(features)
        features = features.reshape(features.shape[0]*features.shape[1], features.shape[2])
        features = np.swapaxes(features,0,1)
        
        return features

    def _resample_signal(self, data, freq):
        
        if freq ==250:
            resampled_signal = data
        else:
            num_of_samples = int(freq*data.shape[0]/250) # freq * numb of seconds
            resampled_signal = signal.resample(data, num_of_samples)

        return resampled_signal
    
    def _transform_data(self, TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls, accel_Or_gyro, Filtering, frequency):
        
        # Extracts only acceleration X,Y,Z and resample the data
        transformed_data = []
        print(len(TUG_one_Sensor_imu))
        for i in range(len(TUG_one_Sensor_imu)):
            
            if (accel_Or_gyro == 'accel & gyro'):
                features_list = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            if (accel_Or_gyro == 'accel'):
                features_list = ['accel_x', 'accel_y', 'accel_z']
            if (accel_Or_gyro == 'gyro'):
                features_list = ['gyro_x', 'gyro_y', 'gyro_z']
            if (accel_Or_gyro == 'accelx & gyrox'):
                features_list = ['accel_x', 'gyro_x']
            if (accel_Or_gyro == 'accely & gyroy'):
                features_list = ['accel_y', 'gyro_y']
            if (accel_Or_gyro == 'accelz & gyroz'):
                features_list = ['accel_z', 'gyro_z']
                
            data = self._selected_features_data(TUG_one_Sensor_imu[i], features_list)
            ##### For example, if features_list = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            ##### then data is ith (accelx,y,z, gyrox,y,z)            
            data = self._resample_signal(data, frequency)     
            
            if Filtering != 'No':
                ##### Low/High pass filtering: signal.butter(order, fc, btype, current freq)    
                fc = 50
                
                data_X = data[:, 0]
                data_Y = data[:, 1]
                data_Z = data[:, 2]

                b, a = signal.butter(5, fc, Filtering, frequency)  ### default output is 'ba' for backwards compatibility
                
                data_X = signal.filtfilt(b, a, data_X)
                data_Y = signal.filtfilt(b, a, data_Y)
                data_Z = signal.filtfilt(b, a, data_Z)
                data = np.concatenate((data_X.reshape(len(data),1), data_Y.reshape(len(data),1), data_Z.reshape(len(data),1)), axis=1)
                frequency = fc   #### The new filtered signal has the frequency of the filtering fc

            transformed_data.append(data)
        
        return transformed_data, labels, subject_no, report_fall, no_of_falls, frequency


    def _Three_sec_cut_data(self, signal, window_size, frequency):

        L = signal.shape[0]
        Three_sec_cut_data = []
        step_length = window_size
        Numb_of_segments = math.ceil((L-window_size)/frequency) +1
        for i in range(Numb_of_segments): 
            mod = (L - i*frequency) % window_size
            if ((i==Numb_of_segments-1) & (mod!=0)):

                signal = np.pad(signal, ((0,window_size-(L - i*frequency)), (0,0)), 'constant', constant_values=((0,0),(0,0)) )
            Three_sec_cut_data.append(signal[ i*frequency : i*frequency + step_length, : ])
        Three_sec_cut_data = np.asarray(Three_sec_cut_data)
      
        return Three_sec_cut_data


    def _separate_faller_and_nonfaller(self, data, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features):
        [age, gender, weight, height, BP] = identity_features 

        # Create empty lists of faller and non-faller raw signals
        F_data, nonF_data = [], []
        F_subject_no, nonF_subject_no = [], []
        F_report_fall, nonF_report_fall = [], []
        F_no_of_falls, nonF_no_of_falls = [], []
        F_composite_score, nonF_composite_score = [], []
        
        F_age, nonF_age = [], []
        F_gender, nonF_gender = [], []
        F_weight, nonF_weight = [], []
        F_height, nonF_height = [], []
        F_BP, nonF_BP = [], []
        
        # Create empty lists of after follow up faller and non-faller raw signals
        Faller_data, nonFaller_data = [], []
        Faller_subject_no, nonFaller_subject_no = [], []
        Faller_no_of_falls, nonFaller_no_of_falls = [], []
        Faller_composite_score, nonFaller_composite_score = [], []
        
        Faller_age, nonFaller_age = [], []
        Faller_gender, nonFaller_gender = [], []
        Faller_weight, nonFaller_weight = [], []
        Faller_height, nonFaller_height = [], []
        Faller_BP, nonFaller_BP = [], []
        
        # Separate diagnosed fallers and non-fallers signals and save them in F_data and nonF_data lists
        for i in range(len(data)):
            if labels[i]==1:
                F_data.append(data[i])
                F_subject_no.append(subject_no[i])
                F_report_fall.append(report_fall[i])
                F_no_of_falls.append(no_of_falls[i])
                F_composite_score.append(composite_score[i])
                F_age.append(age[i])
                F_gender.append(gender[i])
                F_weight.append(weight[i])
                F_height.append(height[i])
                F_BP.append(BP[i])
                
            if labels[i]==0:
                nonF_data.append(data[i])
                nonF_subject_no.append(subject_no[i])
                nonF_report_fall.append(report_fall[i])
                nonF_no_of_falls.append(no_of_falls[i])
                nonF_composite_score.append(composite_score[i])
                nonF_age.append(age[i])
                nonF_gender.append(gender[i])
                nonF_weight.append(weight[i])
                nonF_height.append(height[i])
                nonF_BP.append(BP[i])
                     
        # Separate follow up results fallers and non-fallers signals and save them in F_data and nonF_data lists
        for i in range(len(data)):
            if report_fall[i]==1:
                Faller_data.append(data[i])
                Faller_subject_no.append(subject_no[i])
                Faller_no_of_falls.append(no_of_falls[i])
                Faller_composite_score.append(composite_score[i])
                Faller_age.append(age[i])
                Faller_gender.append(gender[i])
                Faller_weight.append(weight[i])
                Faller_height.append(height[i])
                Faller_BP.append(BP[i])
                
            if report_fall[i]==0:
                nonFaller_data.append(data[i])
                nonFaller_subject_no.append(subject_no[i])
                nonFaller_no_of_falls.append(no_of_falls[i])
                nonFaller_composite_score.append(composite_score[i])
                nonFaller_age.append(age[i])
                nonFaller_gender.append(gender[i])
                nonFaller_weight.append(weight[i])
                nonFaller_height.append(height[i])
                nonFaller_BP.append(BP[i])
                                
        data = [F_data, F_subject_no, F_report_fall, F_no_of_falls, F_composite_score, F_age, F_gender, F_weight, F_height, F_BP, nonF_data, nonF_subject_no, nonF_report_fall, nonF_no_of_falls, nonF_composite_score, nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP]
        followup_data = [Faller_data, Faller_subject_no, Faller_no_of_falls, Faller_composite_score, Faller_age, Faller_gender, Faller_weight, Faller_height, Faller_BP, nonFaller_data, nonFaller_subject_no, nonFaller_no_of_falls, nonFaller_composite_score, nonFaller_age, nonFaller_gender, nonFaller_weight, nonFaller_height, nonFaller_BP]
        
        return data, followup_data

    def _segment_data(self, data, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features, sensors, window_size, frequency):
  
        ## F_patient_label and nonF_patient_label are not used in this function.
        ## They are returned in _separate_faller_and_nonfaller function forthe case of raw signal data but not the segmented data
        RF = window_size*frequency
        data, followup_data = self._separate_faller_and_nonfaller(data, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features)
        
        [F_data, F_subject_no, F_report_fall, F_no_of_falls, F_composite_score, F_age, F_gender, F_weight, F_height, F_BP, nonF_data, nonF_subject_no, nonF_report_fall, nonF_no_of_falls, nonF_composite_score, nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP] = data
        [Faller_data, Faller_subject_no, Faller_no_of_falls, Faller_composite_score, Faller_age, Faller_gender, Faller_weight, Faller_height, Faller_BP, nonFaller_data, nonFaller_subject_no, nonFaller_no_of_falls, nonFaller_composite_score, nonFaller_age, nonFaller_gender, nonFaller_weight, nonFaller_height, nonFaller_BP] = followup_data
     
        F_bag_of_segments, nonF_bag_of_segments = [], []
        F_bag_of_subject_no, nonF_bag_of_subject_no = [], []
        F_bag_of_report_fall, nonF_bag_of_report_fall = [], []
        F_bag_of_no_of_falls, nonF_bag_of_no_of_falls = [], []
        F_bag_of_comp_score, nonF_bag_of_comp_score = [], []
        F_bag_of_seg_no, nonF_bag_of_seg_no = [], []
        F_bag_of_age, nonF_bag_of_age, F_bag_of_gender, nonF_bag_of_gender, F_bag_of_weight, nonF_bag_of_weight = [], [], [], [], [], []
        F_bag_of_height, nonF_bag_of_height, F_bag_of_BP, nonF_bag_of_BP = [], [], [], []
        
        for i in range(len(F_subject_no)):
            F_segments = self._Three_sec_cut_data(F_data[i], RF, frequency)  

            for j in range(F_segments.shape[0]):
                F_bag_of_segments.append(F_segments[j].reshape(1,F_segments[j].shape[0], F_segments[j].shape[1]))
                F_bag_of_subject_no.append(F_subject_no[i])
                F_bag_of_report_fall.append(F_report_fall[i])
                F_bag_of_no_of_falls.append(F_no_of_falls[i])
                F_bag_of_comp_score.append(F_composite_score[i])
                F_bag_of_seg_no.append(j)
                F_bag_of_age.append(F_age[i])
                F_bag_of_gender.append(F_gender[i])
                F_bag_of_weight.append(F_weight[i])
                F_bag_of_height.append(F_height[i])
                F_bag_of_BP.append(F_BP[i])

        for i in range(len(nonF_subject_no)):
            nonF_segments = self._Three_sec_cut_data(nonF_data[i], RF, frequency)
   
            for k in range(nonF_segments.shape[0]):    
                nonF_bag_of_segments.append(nonF_segments[k].reshape(1,nonF_segments[k].shape[0], nonF_segments[k].shape[1]))
                nonF_bag_of_subject_no.append(nonF_subject_no[i])
                nonF_bag_of_report_fall.append(nonF_report_fall[i])
                nonF_bag_of_no_of_falls.append(nonF_no_of_falls[i])
                nonF_bag_of_comp_score.append(nonF_composite_score[i])
                nonF_bag_of_seg_no.append(k)
                nonF_bag_of_age.append(nonF_age[i])
                nonF_bag_of_gender.append(nonF_gender[i])
                nonF_bag_of_weight.append(nonF_weight[i])
                nonF_bag_of_height.append(nonF_height[i])
                nonF_bag_of_BP.append(nonF_BP[i])

        F_bag_of_segments = np.concatenate(F_bag_of_segments)
        nonF_bag_of_segments = np.concatenate(nonF_bag_of_segments)

        data = [F_bag_of_segments, F_bag_of_subject_no, F_bag_of_report_fall, F_bag_of_no_of_falls, F_bag_of_comp_score, F_bag_of_seg_no, F_bag_of_age, F_bag_of_gender, F_bag_of_weight, F_bag_of_height, F_bag_of_BP, nonF_bag_of_segments, nonF_bag_of_subject_no, nonF_bag_of_report_fall, nonF_bag_of_no_of_falls, nonF_bag_of_comp_score, nonF_bag_of_seg_no, nonF_bag_of_age, nonF_bag_of_gender, nonF_bag_of_weight, nonF_bag_of_height, nonF_bag_of_BP]

        ### AFTER FOLLOWUP: Segmentation of data       
        Faller_bag_of_segments, nonFaller_bag_of_segments = [], []
        Faller_bag_of_subject_no, nonFaller_bag_of_subject_no = [], []
        Faller_bag_of_no_of_falls, nonFaller_bag_of_no_of_falls = [], []
        Faller_bag_of_comp_score, nonFaller_bag_of_comp_score = [], []
        Faller_bag_of_seg_no, nonFaller_bag_of_seg_no = [], []
        Faller_bag_of_age, nonFaller_bag_of_age, Faller_bag_of_gender, nonFaller_bag_of_gender, Faller_bag_of_weight, nonFaller_bag_of_weight = [], [], [], [], [], []
        Faller_bag_of_height, nonFaller_bag_of_height, Faller_bag_of_BP, nonFaller_bag_of_BP = [], [], [], []

        for i in range(len(Faller_subject_no)):
            Faller_segments = self._Three_sec_cut_data(Faller_data[i], RF, frequency)  

            for j in range(Faller_segments.shape[0]):
                Faller_bag_of_segments.append(Faller_segments[j].reshape(1,Faller_segments[j].shape[0], Faller_segments[j].shape[1]))
                Faller_bag_of_subject_no.append(Faller_subject_no[i])
                Faller_bag_of_no_of_falls.append(Faller_no_of_falls[i])
                Faller_bag_of_comp_score.append(Faller_composite_score[i])
                Faller_bag_of_seg_no.append(j)
                Faller_bag_of_age.append(Faller_age[i])
                Faller_bag_of_gender.append(Faller_gender[i])
                Faller_bag_of_weight.append(Faller_weight[i])
                Faller_bag_of_height.append(Faller_height[i])
                Faller_bag_of_BP.append(Faller_BP[i])

        for i in range(len(nonFaller_subject_no)):
            nonFaller_segments = self._Three_sec_cut_data(nonFaller_data[i], RF, frequency)
   
            for k in range(nonFaller_segments.shape[0]):    
                nonFaller_bag_of_segments.append(nonFaller_segments[k].reshape(1,nonFaller_segments[k].shape[0], nonFaller_segments[k].shape[1]))
                nonFaller_bag_of_subject_no.append(nonFaller_subject_no[i])
                nonFaller_bag_of_no_of_falls.append(nonFaller_no_of_falls[i])
                nonFaller_bag_of_comp_score.append(nonFaller_composite_score[i])
                nonFaller_bag_of_seg_no.append(k)
                nonFaller_bag_of_age.append(nonFaller_age[i])
                nonFaller_bag_of_gender.append(nonFaller_gender[i])
                nonFaller_bag_of_weight.append(nonFaller_weight[i])
                nonFaller_bag_of_height.append(nonFaller_height[i])
                nonFaller_bag_of_BP.append(nonFaller_BP[i])

        Faller_bag_of_segments = np.concatenate(Faller_bag_of_segments)
        nonFaller_bag_of_segments = np.concatenate(nonFaller_bag_of_segments)       
        followup_data = [Faller_bag_of_segments, Faller_bag_of_subject_no, Faller_bag_of_no_of_falls, Faller_bag_of_comp_score, Faller_bag_of_seg_no, Faller_bag_of_age, Faller_bag_of_gender, Faller_bag_of_weight, Faller_bag_of_height, Faller_bag_of_BP, nonFaller_bag_of_segments, nonFaller_bag_of_subject_no, nonFaller_bag_of_no_of_falls, nonFaller_bag_of_comp_score, nonFaller_bag_of_seg_no, nonFaller_bag_of_age, nonFaller_bag_of_gender, nonFaller_bag_of_weight, nonFaller_bag_of_height, nonFaller_bag_of_BP]
        
        return data, followup_data

    def _zeropad_data(self, data, labels, subject_no, report_fall, no_of_falls):
        
        max_length = np.amax([ data[i].shape[0] for i in range(len(data)) ]) 
        padded_data = []
        for i in range(len(data)):
            padded_data.append(np.pad(data[i], ( (0,max_length - data[i].shape[0]) , (0,0) ), 'constant' ))
            
        padded_data = np.asarray(padded_data)    
        
        return padded_data, labels, subject_no, report_fall, no_of_falls

    ##########################################################################
    def segmented_TUG(self, sensors, accel_Or_gyro, Filtering, frequency, window_size, stride):
        
        [TUG_imu, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features] = self.data
        TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls = self._TUG_oneSensor_data(TUG_imu, labels, subject_no, report_fall, no_of_falls, sensors)
        transformed_data, labels, subject_no, report_fall, no_of_falls, updated_frequency = self._transform_data(TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls, accel_Or_gyro, Filtering, frequency)
        data, followup_data = self._segment_data(transformed_data, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features, sensors, window_size, updated_frequency)
        return data, followup_data

    
    ### raw signal with zero padding ########################################
    def rawsignal_TUG(self, sensors, accel_Or_gyro, Filtering, frequency):
        
        [TUG_imu, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features] = self.data
        TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls = self._TUG_oneSensor_data(TUG_imu, labels, subject_no, report_fall, no_of_falls, sensors)
        transformed_data, labels, subject_no, report_fall, no_of_falls, updated_frequency = self._transform_data(TUG_one_Sensor_imu, labels, subject_no, report_fall, no_of_falls, accel_Or_gyro, Filtering, frequency)
        padded_data, labels, subject_no, report_fall, no_of_falls = self._zeropad_data(transformed_data, labels, subject_no, report_fall, no_of_falls)
        data, followup_data = self._separate_faller_and_nonfaller(padded_data, labels, subject_no, report_fall, no_of_falls, composite_score, identity_features)
                
        [F_data, F_subject_no, F_report_fall, F_no_of_falls, F_composite_score, F_age, F_gender, F_weight, F_height, F_BP, nonF_data, nonF_subject_no, nonF_report_fall, nonF_no_of_falls, nonF_composite_score, nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP] = data
        F_data = np.asarray(F_data)
        F_subject_no = np.asarray(F_subject_no)
        nonF_data = np.asarray(nonF_data)
        nonF_subject_no = np.asarray(nonF_subject_no)
        
        [Faller_data, Faller_subject_no, Faller_no_of_falls, Faller_composite_score, Faller_age, Faller_gender, Faller_weight, Faller_height, Faller_BP, nonFaller_data, nonFaller_subject_no, nonFaller_no_of_falls, nonFaller_composite_score, nonFaller_age, nonFaller_gender, nonFaller_weight, nonFaller_height, nonFaller_BP] = followup_data
        
        
        Faller_data = np.asarray(Faller_data)
        Faller_subject_no = np.asarray(Faller_subject_no)
        nonFaller_data = np.asarray(nonFaller_data)
        nonFaller_subject_no = np.asarray(nonFaller_subject_no)
        
        followup_data = [Faller_data, Faller_subject_no, Faller_no_of_falls, Faller_composite_score, Faller_age, Faller_gender, Faller_weight, Faller_height, Faller_BP, nonFaller_data, nonFaller_subject_no, nonFaller_no_of_falls, nonFaller_composite_score, nonFaller_age, nonFaller_gender, nonFaller_weight, nonFaller_height, nonFaller_BP]
        data = [F_data, F_subject_no, F_report_fall, F_no_of_falls, F_composite_score, F_age, F_gender, F_weight, F_height, F_BP, nonF_data, nonF_subject_no, nonF_report_fall, nonF_no_of_falls, nonF_composite_score, nonF_age, nonF_gender, nonF_weight, nonF_height, nonF_BP]
        return data, followup_data
