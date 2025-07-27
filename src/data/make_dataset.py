import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv('../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_gyr = pd.read_csv('../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files =  glob('../../data/raw/MetaMotion/*.csv')
len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
f = files[0]

participant = f.split('-')[0][-1]

label= f.split('-')[1]

category = f.split('-')[2].rstrip('1234').rstrip('_MetaWear_2019')

df = pd.read_csv(f)

df['participant'] = participant
df['label'] = label
df['category'] = category

# if 'Gyroscope' in  f :
#     print('Yes')



# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# This will be the key for every set (file) 
acc_set = 1
gyr_set = 1

for f in files:
    
    participant = f.split('-')[0][-1]

    label= f.split('-')[1]

    category = f.split('-')[2].rstrip('1234').rstrip('_MetaWear_2019')
    
    df = pd.read_csv(f)

    df['participant'] = participant
    df['label'] = label
    df['category'] = category
    
    if 'Accelerometer' in f:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])  # Pass the DataFrames as elements of a list
    
    if 'Gyroscope' in  f :
        df['set'] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])  # Pass the DataFrames as elements of a list

    




# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

pd.to_datetime(acc_df['epoch (ms)'] , unit='ms') # epoch is on Unix time format

pd.to_datetime(acc_df['time (01:00)'])

#set index as date time
acc_df.index = pd.to_datetime(acc_df['epoch (ms)'] , unit='ms')
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'] , unit='ms')

# Drop any columns related to date or time 
acc_df.drop(['epoch (ms)' , 'time (01:00)' , 'elapsed (s)'] , axis=1 , inplace=True)
gyr_df.drop(['epoch (ms)' , 'time (01:00)' , 'elapsed (s)'] , axis=1 , inplace=True)



# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files =  glob('../../data/raw/MetaMotion/*.csv')

def read_data_from_files(files):
    """
    Read data from a list of files and concatenate accelerometer and gyroscope data.

    Parameters:
    - files (list): A list of file paths containing data.

    Returns:
    - acc_df (DataFrame): A DataFrame containing concatenated accelerometer data.
    - gyr_df (DataFrame): A DataFrame containing concatenated gyroscope data.

    This function reads data from a list of files, where each file is expected to contain 
    accelerometer or gyroscope data. It concatenates the data based on the type (Accelerometer 
    or Gyroscope) and assigns a 'set' value to each set of data. It then sets the index of 
    each DataFrame to the epoch time and drops unnecessary columns related to date and time.
    """
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # This will be the key for every set (file) 
    acc_set = 1
    gyr_set = 1

    for f in files:
        
        participant = f.split('-')[0][-1]
        label= f.split('-')[1]
        category = f.split('-')[2].rstrip('1234').rstrip('_MetaWear_2019')
        
        df = pd.read_csv(f)

        df['participant'] = participant
        df['label'] = label
        df['category'] = category
        
        if 'Accelerometer' in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])  # Pass the DataFrames as elements of a list
        
        if 'Gyroscope' in  f :
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])  # Pass the DataFrames as elements of a list
            
    # Set index as date time
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'] , unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'] , unit='ms')

    # Drop any columns related to date or time 
    acc_df.drop(['epoch (ms)' , 'time (01:00)' , 'elapsed (s)'] , axis=1 , inplace=True)
    gyr_df.drop(['epoch (ms)' , 'time (01:00)' , 'elapsed (s)'] , axis=1 , inplace=True)
        
    return acc_df , gyr_df

    
acc_df , gyr_df =read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


data_merged =  pd.concat([acc_df.iloc[:,:3] , gyr_df]  , axis=1)

data_merged.columns = ['acc_x',
                       'acc_y',
                       'acc_z',
                       'gyr_x',
                       'gyr_y',
                       'gyr_z', 
                       'participant', 
                       'label', 
                       'category',
                       'set']

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

#The Gyroscope is faster than the Accelerometer, so we need to Unifying the time between each measurement and the other in both sensors

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_merged[:1000].resample(rule='200ms').apply(sampling)
#This means every 200ms, we take all measurement and put them into one record by taking the mean for them

#but if making that in hole data set at same time this message will display: MemoryError: Unable to allocate 210. MiB for an array with shape (7, 3932145) and data type float64
#so we will divide the df into small groups by grouping them by days and then concatenate again
days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
days[0]

data_resampled =  pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

data_resampled['set'] = data_resampled['set'].astype('int')
data_resampled.info()
#now we have final data 

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle('../../data/interim/01_data_processed.pkl')