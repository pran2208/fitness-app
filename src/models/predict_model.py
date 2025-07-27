"""
This module contains the implementation of a Fitness Tracker Predictor class
used for predicting activities based on accelerometer and gyroscope data.

The FitnessTrackerPredictor class performs various tasks including:
- Reading accelerometer and gyroscope data from CSV files
- Removing outliers from the data frame
- Applying feature engineering techniques such as: 
        low-pass filtering,
        PCA,
        temporal and frequency abstraction
- Predicting activity using a trained model
- Counting repetitions based on specific activity labels

The module also imports necessary libraries and modules for data manipulation,
visualization, and machine learning tasks.

"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.signal import argrelextrema
from src.models.outliers_remove import mark_outliers_chauvenet
from src.models.DataTransformation import LowPassFilter , PrincipalComponentAnalysis
from src.models.TemporalAbstraction import NumericalAbstraction
from src.models.FrequencyAbstraction import FourierTransformation
warnings.filterwarnings("ignore")
# from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn


class FitnessTrackerPredictor:
    """
    A class for predicting activities based on accelerometer and gyroscope data.

    This class provides methods to read data from CSV files, preprocess the data
    by removing outliers and applying feature engineering techniques such as low-pass
    filtering, PCA, temporal and frequency abstraction. It also includes methods
    for predicting activity using a trained model and counting repetitions based on
    specific activity labels.

    Attributes:
        acc_path (str): The file path of the accelerometer data CSV file.
        gyr_path (str): The file path of the gyroscope data CSV file.
        model_path (str): The file path of the trained model for activity prediction.
        cluster_model_path (str): The file path of the trained clustering model.
    """

    def __init__(self, acc_path, gyr_path, model_path, cluster_model_path):
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

        pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        pd.to_datetime(acc_df["time (01:00)"])

        acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
        gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

        acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
        gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)

        data_merged = pd.concat([acc_df, gyr_df], axis=1)
        data_merged.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

        sampling = {
            "acc_x": "mean",
            "acc_y": "mean",
            "acc_z": "mean",
            "gyr_x": "mean",
            "gyr_y": "mean",
            "gyr_z": "mean",
        }

        data_merged[:1000].resample(rule="200ms").apply(sampling)
        # Every 200ms, we average measurements to create one record.

        # If making that in hole data set at same time,
        # this message will display: MemoryError: Unable to allocate 210.
        # MiB for an array with shape (7, 3932145) and data type float64
        # To handle this, we split the dataframe into smaller groups by day, then concatenate them.


        days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
        #days[0]

        data_resampled = pd.concat(
            [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
        )

        return data_resampled

    def remove_outliers(self):
        """
        Remove outliers from the data frame.
        """
        outliers_removed_df = self.read_data()
        for col in outliers_removed_df:
            dataset = mark_outliers_chauvenet(outliers_removed_df, col=col)
            dataset.loc[dataset[col + "_outlier"], col] = np.nan
            outliers_removed_df[col] = dataset[col].interpolate()
        return outliers_removed_df

    def apply_feature_engineering(self):
        """
        Apply feature engineering to the data frame.
        """
        df_lowpass = self.remove_outliers()
        sensor_col = list(df_lowpass.columns)

        low_pass = LowPassFilter()

        sampling_frq = (
            1000 / 200
        )
        # As data is recorded every 200 ms, this line computes the number of repetitions per second.


        cutoff_frq = 1.3  # the low cutoff frequency, meaning more smoothing in signal

        # LowPass.low_pass_filter(df_lowpass , 'acc_y' , sampling_frq , cutoff_frq)

        for col in df_lowpass.columns:
            df_lowpass = low_pass.low_pass_filter(
                df_lowpass, col, sampling_frq, cutoff_frq, order=5
            )
            df_lowpass[col] = df_lowpass[col + "_lowpass"]
            del df_lowpass[col + "_lowpass"]

        print("in low pass filter:", len(df_lowpass))
        # ----------------PCA----------------

        df_pca = df_lowpass.copy()
        pca = PrincipalComponentAnalysis()

        df_pca = pca.apply_pca(df_pca, df_lowpass.columns, 3)

        # ----------------squares----------------

        df_squares = df_pca.copy()

        acc_r = (
            df_squares["acc_x"] ** 2
            + df_squares["acc_y"] ** 2
            + df_squares["acc_z"] ** 2
        )
        gyr_r = (
            df_squares["gyr_x"] ** 2
            + df_squares["gyr_y"] ** 2
            + df_squares["gyr_z"] ** 2
        )

        df_squares["acc_r"] = np.sqrt(acc_r)
        df_squares["gyr_r"] = np.sqrt(gyr_r)

        print("in low PCA:", len(df_squares))

        # ------------------------------------temporal ------------------------------------
        df_temporal = df_squares.copy()
        sensor_col = sensor_col + ["acc_r", "gyr_r"]
        #NumAbs = NumericalAbstraction()
        num_abs = NumericalAbstraction()

        # NumAbs.abstract_numerical(df_temporal, sensor_col,
        #                           window_size=5, aggregation_function = 'mean')
        # Moving averages are required for each set due to potential label differences on each set.


        # df_temporal_list = []
        for col in sensor_col:
            subset = num_abs.abstract_numerical(
                df_temporal, sensor_col, window_size=5, aggregation_function="mean"
            )
            subset = num_abs.abstract_numerical(
                subset, sensor_col, window_size=5, aggregation_function="std"
            )

        df_temporal = subset

        print("in low PCA:", len(df_temporal))

        # -----------------------------------FourierTransformation-----------------------------

        df_frq = df_temporal.copy().reset_index()
        # df_frq
        #FreqAbd = FourierTransformation()
        freq_abd = FourierTransformation()

        sampling_frq = int(1000 / 200)
        window_size = int(2800 / 200)

        subset = df_frq.reset_index(drop=True).copy()

        subset = freq_abd.abstract_frequency(
            subset, sensor_col, window_size, sampling_frq
        )

        df_frq = subset.set_index("epoch (ms)", drop=True)

        # --------------------------------------------------------------
        # Dealing with overlapping windows
        # --------------------------------------------------------------

        # All extra features are based on moving averages,
        # so the value between the different rows are highly correlated
        # And this could cause overfitting in our model

        # So we need dealing with that: we will take 50% from data by skipping one row in each step
        df_frq = df_frq.dropna()
        df_frq = df_frq.iloc[::2]

        # Add clustering
        kmeans = joblib.load(self.cluster_model_path)
        cluster_col = ["acc_x", "acc_y", "acc_z"]
        subset = df_frq[cluster_col]
        df_frq["cluster"] = kmeans.predict(subset)
        df_frq["cluster"] = df_frq["cluster"].value_counts().index[0]

        return df_frq

    def predict_activity(self):
        """

        Predict activity using the trained model.

        """
        data_frame = self.apply_feature_engineering()
        feature_set = [
            "acc_y_temp_std_ws_5",
            "gyr_z_temp_std_ws_5",
            "acc_y_freq_0.0_Hz_ws_14",
            "acc_y_freq_0.714_Hz_ws_14",
            "acc_x_freq_0.0_Hz_ws_14",
            "acc_x_freq_0.714_Hz_ws_14",
            "acc_z_freq_0.0_Hz_ws_14",
            "gyr_z_freq_0.714_Hz_ws_14",
            "gyr_r_freq_0.0_Hz_ws_14",
            "gyr_r_freq_0.357_Hz_ws_14",
        ]
        data_frame = data_frame[feature_set]
        # print(data_frame)
        model = joblib.load(self.model_path)
        # data_frame.columns = [col.replace('_lowpass', '') for col in data_frame.columns]
        pred = model.predict(data_frame)
        return pd.DataFrame(pred).mode()[0][0]

    def count_repetitions(self, label):
        """
        Count repetitions based on specific label.
        """
        data_frame = self.read_data()

        # label = self.predict_activity()
        def count_reps_helper(dataset, cutoff=0.4, order=10, column="acc_x"):
            dataset = dataset.copy()

            # Sum of squares attributes
            acc_r = (
                dataset["acc_x"] ** 2 + dataset["acc_y"] ** 2 + dataset["acc_z"] ** 2
            )
            gyr_r = (
                dataset["gyr_x"] ** 2 + dataset["gyr_y"] ** 2 + dataset["gyr_z"] ** 2
            )
            dataset["acc_r"] = np.sqrt(acc_r)
            dataset["gyr_r"] = np.sqrt(gyr_r)

            freq_sampling = 1000 / 200
            lowpass = LowPassFilter()
            data = lowpass.low_pass_filter(
                dataset,
                col=column,
                sampling_frequency=freq_sampling,
                cutoff_frequency=cutoff,
                order=order,
            )
            indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
            peaks = data.iloc[indexes]

            plt.figure(figsize=(15, 5))
            plt.plot(data[column + "_lowpass"])
            plt.plot(peaks[column + "_lowpass"], "o", color="red")
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.savefig(
                r"D:\Programing\Projects\Fitness-tracker-based-on-ML-2\static\pred\count_rep.png"
            )
            # path = '/static/pred/count_rep.jpg'

            return len(peaks)

        if label == "squat":
            cutoff = 0.4
            column = "acc_r"
        elif label == "row":
            cutoff = 0.7
            column = "gyr_r"
        elif label == "ohp":
            cutoff = 0.5
            column = "acc_y"
        elif label == "dead":
            cutoff = 0.5
            column = "acc_y"
        else:
            column = "gyr_y"
            cutoff = 0.5

        return count_reps_helper(data_frame, column=column, cutoff=cutoff)
