import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn
from src.models.outliers_remove import mark_outliers_chauvenet
from src.models.DataTransformation import LowPassFilter , PrincipalComponentAnalysis
from src.models.TemporalAbstraction import NumericalAbstraction
from src.models.FrequencyAbstraction import FourierTransformation
import math
import joblib
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans



class FitnessTrackerPredictor:
    """
    A class for predicting fitness activities based on accelerometer and gyroscope data.

    Parameters:
    acc_path (str): File path for the accelerometer data.
    gyr_path (str): File path for the gyroscope data.
    model_path (str): File path for the trained prediction model.
    cluster_model_path (str): File path for the trained clustering model.

    Attributes:
    acc_path (str): File path for the accelerometer data.
    gyr_path (str): File path for the gyroscope data.
    model_path (str): File path for the trained prediction model.
    cluster_model_path (str): File path for the trained clustering model.

    Methods:
    read_data(): Read accelerometer and gyroscope data from CSV files and merge them.
    remove_outliers(): Remove outliers from the data frame.
    apply_feature_engineering(): Apply feature engineering to the data frame.
    predict_activity(): Predict activity using the trained model.
    count_repetitions(label): Count repetitions based on specific label.
    """
    def __init__(self, acc_path, gyr_path, model_path, cluster_model_path):
        """
        Initialize FitnessTrackerPredictor with file paths for accelerometer, gyroscope data,
        trained prediction model, and trained clustering model.
        """
        self.acc_path = acc_path
        self.gyr_path = gyr_path
        self.model_path = model_path
        self.cluster_model_path = cluster_model_path

    def read_data(self):
        """
        Read accelerometer and gyroscope data from CSV files and merge them.
        """
        acc_df = pd.read_csv(self.acc_path)
        gyr_df = pd.read_csv(self.gyr_path)

        pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
        pd.to_datetime(acc_df['time (01:00)'])

        acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
        gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

        acc_df.drop(['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis=1, inplace=True)
        gyr_df.drop(['epoch (ms)', 'time (01:00)', 'elapsed (s)'], axis=1, inplace=True)

        data_merged = pd.concat([acc_df, gyr_df], axis=1)
        data_merged.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        
        
        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
        }

        data_merged[:1000].resample(rule='200ms').apply(sampling)
        #This means every 200ms, we take all measurement and put them into one record by taking the mean for them

        #but if making that in hole data set at same time this message will display: MemoryError: Unable to allocate 210. MiB for an array with shape (7, 3932145) and data type float64
        #so we will divide the df into small groups by grouping them by days and then concatenate again
        days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
        days[0]

        data_resampled =  pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

        return data_resampled

    def remove_outliers(self):
        """
        Remove outliers from the data frame.
        """
        outliers_removed_df = self.read_data()
        for col in outliers_removed_df:
            dataset = mark_outliers_chauvenet(outliers_removed_df, col=col)
            dataset.loc[dataset[col+'_outlier'], col] = np.nan
            outliers_removed_df[col] = dataset[col].interpolate()
        return outliers_removed_df

    def apply_feature_engineering(self):
        """
        Apply feature engineering to the data frame.
        """
        df_lowpass = self.remove_outliers()
        sensor_col = list(df_lowpass.columns)

        LowPass = LowPassFilter()

        sampling_frq = 1000 / 200 # # because we are taking the record every 200 ms, so that line calculates number of repetition in 1 sec

        cutoff_frq = 1.3 # the low cutoff frequency, meaning more smoothing in signal

        # LowPass.low_pass_filter(df_lowpass , 'acc_y' , sampling_frq , cutoff_frq)



        for col in df_lowpass.columns:
            df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling_frq, cutoff_frq, order=5)
            df_lowpass[col] = df_lowpass[col + "_lowpass"]
            del df_lowpass[col + "_lowpass"]
            
        print('in low pass filter:' ,  len(df_lowpass))
        #---------------------------------------PCA-------------------------------------------------------
            
            
        df_pca = df_lowpass.copy()
        PCA = PrincipalComponentAnalysis()

        df_pca = PCA.apply_pca(df_pca , df_lowpass.columns , 3 )
        
        #-----------------------------------squares------------------------------------------------------

        df_squares = df_pca.copy()

        acc_r = df_squares['acc_x'] **2 + df_squares['acc_y'] **2 +df_squares['acc_z'] **2 
        gyr_r = df_squares['gyr_x'] **2 + df_squares['gyr_y'] **2 +df_squares['gyr_z'] **2 


        df_squares['acc_r'] = np.sqrt(acc_r)
        df_squares['gyr_r'] = np.sqrt(gyr_r)

        print('in low PCA:' , len(df_squares))



        # ------------------------------------temporal ------------------------------------
        df_temporal =  df_squares.copy() 
        sensor_col = sensor_col + ['acc_r' , 'gyr_r']
        NumAbs = NumericalAbstraction()

        # NumAbs.abstract_numerical(df_temporal , sensor_col , window_size=5 ,aggregation_function= 'mean' )
        # we need to make moving average on each set because each set may containing different label (exercise)


        df_temporal_list = []
        for col in sensor_col:
            subset = NumAbs.abstract_numerical(df_temporal , sensor_col , window_size=5 ,aggregation_function= 'mean' )
            subset = NumAbs.abstract_numerical(subset , sensor_col , window_size=5 ,aggregation_function= 'std' )


        df_temporal =  subset

        print('in low PCA:' , len(df_temporal))
        
        # -----------------------------------FourierTransformation-----------------------------------

        df_frq = df_temporal.copy().reset_index()
        # df_frq
        FreqAbd = FourierTransformation()

        sampling_frq = int(1000 / 200)
        window_size = int (2800 / 200)



        subset = df_frq.reset_index(drop = True).copy()
            
        subset =FreqAbd.abstract_frequency(subset , sensor_col , window_size , sampling_frq)
            

        df_frq =  subset.set_index('epoch (ms)' , drop=True)


        # --------------------------------------------------------------
        # Dealing with overlapping windows
        # --------------------------------------------------------------

        # All extra features are based on moving averages, so the value between the different rows are highly correlated #And this could cause overfitting in our model

        # So we need dealing with that: we will take 50% from data by skipping one row in each step
        df_frq
        df_frq = df_frq.dropna()
        df_frq = df_frq.iloc[: :2]

       
        # Add clustering
        kmeans = joblib.load(self.cluster_model_path)
        cluster_col = ['acc_x', 'acc_y', 'acc_z']
        subset = df_frq[cluster_col]
        df_frq['cluster'] = kmeans.predict(subset)
        df_frq['cluster'] = df_frq['cluster'].value_counts().index[0]

        return df_frq

    def predict_activity(self):
        """
        
        Predict activity using the trained model.
        
        """
        data_frame = self.apply_feature_engineering()
        feature_Set= ['acc_y_temp_std_ws_5',
        'gyr_z_temp_std_ws_5',
        'acc_y_freq_0.0_Hz_ws_14',
        'acc_y_freq_0.714_Hz_ws_14',
        'acc_x_freq_0.0_Hz_ws_14',
        'acc_x_freq_0.714_Hz_ws_14',
        'acc_z_freq_0.0_Hz_ws_14',
        'gyr_z_freq_0.714_Hz_ws_14',
        'gyr_r_freq_0.0_Hz_ws_14',
        'gyr_r_freq_0.357_Hz_ws_14']
        data_frame = data_frame[feature_Set]
        print(data_frame)
        model = joblib.load(self.model_path)
        # data_frame.columns = [col.replace('_lowpass', '') for col in data_frame.columns]
        pred = model.predict(data_frame)
        return pd.DataFrame(pred).mode()[0][0]

    def count_repetitions(self , label ):
        """
        Count repetitions based on specific label.

        Parameters:
        label (str): The label indicating the type of activity (e.g., 'squats', 'rows', 'ohp', 'other').

        Returns:
        int: Number of repetitions detected for the specified activity label.
        """
        data_frame = self.read_data()
        # label = self.predict_activity()
        def count_reps_helper(dataset, cutoff=0.4, order=10, column='acc_x'):
            data = dataset.copy()
            fs = 1000 / 200
            lowpass = LowPassFilter()
            data = lowpass.low_pass_filter(data, col=column, sampling_frequency=fs, cutoff_frequency=cutoff,
                                            order=order)
            indexes = argrelextrema(data[column + '_lowpass'].values, np.greater)
            peaks = data.iloc[indexes]
            
            # plt.plot(data[column + '_lowpass'])
            # plt.plot(peaks[column + '_lowpass'] , 'o' , color = 'red')
            
            return len(peaks)

        if label == 'squar':
            cutoff = 0.35
        elif label == 'row':
            cutoff = 0.65
            column = 'gyr_x'
        elif label == 'ohp':
            cutoff = 0.35
        elif label == 'ohp':
            cutoff = 0.8
        else:
            column = 'acc_x'
            cutoff = 0.4

        return count_reps_helper(data_frame, column=column, cutoff=cutoff)




