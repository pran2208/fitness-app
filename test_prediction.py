# Inside test_prediction.py
from src.models.predict_model import FitnessTrackerPredictor
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Usage Example
acc_path = "data/raw/MetaMotion/E-squat-heavy_MetaWear_2019-01-15T20.14.03.633_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
gyr_path = "data/raw/MetaMotion/E-squat-heavy_MetaWear_2019-01-15T20.14.03.633_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"

acc_df = pd.read_csv(acc_path)
gyr_df = pd.read_csv(gyr_path)


model_path = "models/final_model.pkl"
cluster_model_path = "models/Clustering_model.pkl"

tracker_predictor = FitnessTrackerPredictor(
    acc_path, gyr_path, model_path, cluster_model_path
)


print("\n".join(tracker_predictor.apply_feature_engineering().columns))


# ----------------------------------------------------------------------------------------------------
# ------------------------ test 1 -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def test_read_data():
    # test the expected columns from read_data function
    actual = list(tracker_predictor.read_data().columns)
    expected = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    message = "read_data returned the columns {0} instead of {1}".format(
        actual, expected
    )
    assert actual == expected, message

    # test the expected number of rows from read_data function
    actual = tracker_predictor.read_data().shape[0]
    expected = acc_df.shape[0]
    message = "read_data returned {0} rows, expected less than {1}".format(
        actual, expected
    )
    assert actual < expected, message


# ----------------------------------------------------------------------------------------------------
# ------------------------ test 2 -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def test_remove_outliers():
    # test the expected number of rows from read_data function
    actual = tracker_predictor.remove_outliers().shape[0]
    expected = tracker_predictor.read_data().shape[0]
    message = (
        "remove_outliers returned {0} rows, expected less than or equal {1}".format(
            actual, expected
        )
    )
    assert actual <= expected, message


# ----------------------------------------------------------------------------------------------------
# ------------------------ test 3 -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def test_apply_feature_engineering():
    feature_engineering_columns = set(
        tracker_predictor.apply_feature_engineering().columns
    )
    # test the expected PCA columns
    pca_columns = {"pca_1", "pca_2", "pca_3"}
    message = "apply_feature_engineering didn't return the expected PCA features"
    assert pca_columns.issubset(feature_engineering_columns), message

    # test the expected squares columns
    squares_columns = {"acc_r", "gyr_r"}
    message = "apply_feature_engineering didn't return the expected squares features"
    assert squares_columns.issubset(feature_engineering_columns), message

    # test the expected temporal columns
    temporal_columns = {
        "acc_x_temp_mean_ws_5",
        "acc_x_temp_std_ws_5",
        "gyr_x_temp_mean_ws_5",
        "gyr_x_temp_std_ws_5",
    }
    message = "apply_feature_engineering didn't return the expected temporal features"
    assert temporal_columns.issubset(feature_engineering_columns), message

    # test the expected frequency columns
    frequency_columns = {
        "acc_x_max_freq",
        "acc_x_freq_weighted",
        "acc_x_pse",
        "acc_x_freq_0.0_Hz_ws_14",
        "acc_x_freq_0.357_Hz_ws_14",
        "acc_x_freq_0.714_Hz_ws_14",
        "acc_x_freq_1.071_Hz_ws_14",
        "acc_x_freq_1.429_Hz_ws_14",
        "acc_x_freq_1.786_Hz_ws_14",
        "acc_x_freq_2.143_Hz_ws_14",
        "acc_x_freq_2.5_Hz_ws_14",
    }
    message = "apply_feature_engineering didn't return the expected frequency features"
    assert frequency_columns.issubset(feature_engineering_columns), message

    # test the expected frequency columns
    cluster_column = {"cluster"}
    message = "apply_feature_engineering didn't return the expected cluster features"
    assert cluster_column.issubset(feature_engineering_columns), message


# ----------------------------------------------------------------------------------------------------
# ------------------------ test 4 -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def test_predict_activity():
    labels = ["bench", "dead", "ohp", "row", "squat", "rest"]

    # test the bench label
    bench_acc_path = "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    bench_gyr_path = "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message

    # test the dead label
    dead_acc_path = "data/raw/MetaMotion/A-dead-heavy_MetaWear_2019-01-15T20.35.27.174_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    dead_gyr_path = "data/raw/MetaMotion/A-dead-heavy_MetaWear_2019-01-15T20.35.27.174_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message

    # test the ohp label
    ohp_acc_path = "data/raw/MetaMotion/A-ohp-heavy1-rpe8_MetaWear_2019-01-11T16.38.54.580_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    ohp_gyr_path = "data/raw/MetaMotion/A-ohp-heavy1-rpe8_MetaWear_2019-01-11T16.38.54.580_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message

    # test the row label
    row_acc_path = "data/raw/MetaMotion/A-row-heavy_MetaWear_2019-01-14T15.04.06.123_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    row_gyr_path = "data/raw/MetaMotion/A-row-heavy_MetaWear_2019-01-14T15.04.06.123_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message

    # test the squat label
    squat_acc_path = "data/raw/MetaMotion/A-squat-heavy_MetaWear_2019-01-15T20.04.08.637_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    squat_gyr_path = "data/raw/MetaMotion/A-squat-heavy_MetaWear_2019-01-15T20.04.08.637_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message

    # test the rest label
    rest_acc_path = "data/raw/MetaMotion/A-rest-sitting_MetaWear_2019-01-18T18.22.25.565_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    rest_gyr_path = "data/raw/MetaMotion/A-rest-sitting_MetaWear_2019-01-18T18.22.25.565_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    message = f"predict_activity return {predicted_label} and didn't return any of the expected labels {labels}"
    assert predicted_label in labels, message


# ----------------------------------------------------------------------------------------------------
# ------------------------ test 5 -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def test_count_repetitions():
    # test count repetitions func
    bench_acc_path = "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    bench_gyr_path = "data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    tracker_predictor = FitnessTrackerPredictor(
        bench_acc_path, bench_gyr_path, model_path, cluster_model_path
    )
    predicted_label = tracker_predictor.predict_activity()
    num_rep = tracker_predictor.count_repetitions(predicted_label)
    message = f"count_repetitions return {predicted_label} and expected to return number between 0 and 25."
    assert (num_rep > 0) and (num_rep < 25), message
