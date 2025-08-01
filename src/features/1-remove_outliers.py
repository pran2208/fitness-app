import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/01_data_processed.pkl')


sensor_col = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

mpl.style.use('fivethirtyeight')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

df[['acc_x' , 'label']].boxplot(by ='label' )

df[sensor_col[:3]  + ['label']].boxplot(by ='label' , figsize = (20, 10)  , layout = (1,3))
df[sensor_col[3:]  + ['label']].boxplot(by ='label' , figsize = (20, 10)  , layout = (1,3))

plt.show()

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column

col = 'acc_x'
dataset = mark_outliers_iqr(df , col )
plot_binary_outliers(dataset = dataset , col = col , outlier_col = col+'_outlier' , reset_index=True)


# Loop over all columns

for col in sensor_col:
    dataset = mark_outliers_iqr(df , col )
    plot_binary_outliers(dataset=dataset , col=col , outlier_col=col+'_outlier' , reset_index=True)


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution

sns.kdeplot(df['acc_x'] )

df[sensor_col[:3]  + ['label']].plot.hist(by ='label' , figsize = (20, 10)  , layout = (3,3))
df[sensor_col[3:]  + ['label']].plot.hist(by ='label' , figsize = (20, 10)  , layout = (3,3))

# Insert Chauvenet's function

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    # print('low is:' , low[0])
    # print('high is:' , low.iloc[0])
    
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
            # 1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

col = 'acc_x'
dataset = mark_outliers_chauvenet(df , col )

# Loop over all columns

for col in sensor_col:
    dataset = mark_outliers_chauvenet(df , col )
    plot_binary_outliers(dataset=dataset , col=col , outlier_col=col+'_outlier' , reset_index=True)



# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns

dataset, outliers, X_scores = mark_outliers_lof(df , sensor_col )

for col in sensor_col:
    plot_binary_outliers(dataset=dataset , col=col , outlier_col='outlier_lof' , reset_index=True)



# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label = 'bench'
for col in sensor_col:
    dataset = mark_outliers_iqr(df[df['label'] == label] , col )
    plot_binary_outliers(dataset=dataset , col=col , outlier_col=col+'_outlier' , reset_index=True)

for col in sensor_col:
    dataset = mark_outliers_chauvenet(df[df['label'] == label] , col )
    plot_binary_outliers(dataset=dataset , col=col , outlier_col=col+'_outlier' , reset_index=True)


dataset, outliers, X_scores = mark_outliers_lof(df[df['label'] == label] , sensor_col )

for col in sensor_col:
    plot_binary_outliers(dataset=dataset , col=col , outlier_col='outlier_lof' , reset_index=True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# We tried 3 method 
# 1) The IQR is very sensitive, and it mark a large number of rows as outliers 
# 2) Chauvenet is less sensitive and depends on data distribution
# 3) iof its measure if the point has  number of neighbors beside it or not, if not, it will be an outlier.


# We will use Chauvenet for dealing with outliers

# Test on single column

col = 'gyr_z'

dataset =  mark_outliers_chauvenet(df , col=col)

dataset[dataset['gyr_z_outlier']]

dataset.loc[dataset['gyr_z_outlier'] , 'gyr_z'] = np.nan


# Create a loop

outliers_removed_df = df.copy()

# outliers_removed_df = pd.DataFrame(columns=df.columns)

for col in sensor_col:
    dfs_to_concat = []  # List to collect DataFrames generated within the loop
    for label in df['label'].unique():
        dataset = mark_outliers_chauvenet(df[df['label'] == label], col=col)

        # Replace values marked as outliers with NaN
        dataset.loc[dataset[col+'_outlier'], col] = np.nan

        #Update the column in the orignal dataframe
        outliers_removed_df.loc[(outliers_removed_df['label'] == label) , col] = dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f'Removed {n_outliers} , from {col} for {label}')
        
        

outliers_removed_df.info()
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle('../../data/interim/02_outliers_removed_chauvenets.pkl')
