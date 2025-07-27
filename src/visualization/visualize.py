import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/01_data_processed.pkl')

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

#We plotting single set because if we were plotting all sets, the plot wouldn't be useful and would not be clear to interpret
single_set =  df[df['set'] == 2]

# df['set'].unique()

plt.plot(single_set['acc_y'])

plt.plot(single_set['acc_y'].reset_index(drop= True)) #It tells us how many samples were in the set 




# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
# df['label'].unique()

for label in df['label'].unique():
    subset = df[df['label'] == label]
    # display(subset.head(2))
    fig ,ax =plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop= True) , label=label)
    plt.legend()
    plt.show()
    
for label in df['label'].unique():
    subset = df[df['label'] == label]
    # display(subset.head(2))
    fig ,ax =plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop= True) , label=label)
    plt.legend()
    plt.show()
    
# by show those plots, we note that there are different pattern for each exercise, which is good for our ML model

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat' and participant == 'A' ").reset_index()


category_df.groupby('category')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend()

# we notes that the medium has more frequency than the heavy exercises in accelerometer sensor

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values('participant').reset_index()

participant_df.groupby('participant')['acc_y'].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis (X, Y ,Z)
# --------------------------------------------------------------

label = 'squat'
participant = 'A'

all_axis_df = df.query(f" label == '{label}' and participant == '{participant}' ").reset_index()

all_axis_df[['acc_x' , 'acc_y' , 'acc_z']].plot()
plt.xlabel('samples')
plt.ylabel('acc_y')
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(f" label == '{label}' and participant == '{participant}' ").reset_index()
        
        if len(all_axis_df) > 0: # To fillter the empty df (some particpant didn't make some exercises)

            all_axis_df[['acc_x' , 'acc_y' , 'acc_z']].plot()
            plt.xlabel('samples')
            plt.ylabel('acc_y')
            plt.title(f'{label} , {participant}'.title())
            plt.legend()

#now do it for Gyroscope
            
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f" label == '{label}' and participant == '{participant}' ").reset_index()
        
        if len(all_axis_df) > 0: # To fillter the empty df (some particpant didn't make some exercises)

            all_axis_df[['gyr_x' , 'gyr_y' , 'gyr_z']].plot()
            plt.xlabel('samples')
            plt.ylabel('gyr_y')
            plt.title(f'{label} , {participant}'.title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = 'row'
participant = 'A'

combined_plot_df = df.query(f" label == '{label}' and participant == '{participant}' ").reset_index(drop=True)

fig , ax = plt.subplots(nrows = 2 ,sharex=True , figsize = (20,10) )

combined_plot_df[['acc_x' , 'acc_y' , 'acc_z']].plot(ax=ax[0])
combined_plot_df[['gyr_x' , 'gyr_y' , 'gyr_z']].plot(ax=ax[1])

ax[0].legend(loc = 'upper center' , bbox_to_anchor = (0.5 , 1.15) , ncol = 3 , fancybox = True , shadow = True)
ax[1].legend(loc = 'upper center' , bbox_to_anchor = (0.5 , 1.15) , ncol = 3 , fancybox = True , shadow = True)

ax[1].set_xlabel('samples')

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        
        combined_plot_df = df.query(f" label == '{label}' and participant == '{participant}' ").reset_index(drop=True)
        
        
        if len(combined_plot_df) > 0: # To fillter the empty df (some particpant didn't make some exercises)

            fig , ax = plt.subplots(nrows = 2 ,sharex=True , figsize = (20,10) )

            combined_plot_df[['acc_x' , 'acc_y' , 'acc_z']].plot(ax=ax[0])
            combined_plot_df[['gyr_x' , 'gyr_y' , 'gyr_z']].plot(ax=ax[1])

            ax[0].legend(loc = 'upper center' , bbox_to_anchor = (0.5 , 1.15) , ncol = 3 , fancybox = True , shadow = True)
            ax[1].legend(loc = 'upper center' , bbox_to_anchor = (0.5 , 1.15) , ncol = 3 , fancybox = True , shadow = True)

            ax[1].set_xlabel('samples')
            
            plt.savefig(f'../../reports/figures/{label.title()}_({participant}).png')
            
            plt.show()