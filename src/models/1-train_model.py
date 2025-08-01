import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df =  pd.read_pickle('../../data/interim/03_data_features.pkl')
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df.iloc[: , :10].info()

df.describe().T

df_train =  df.drop(['participant' , 'category' , 'set'] , axis= 1)

X = df_train.drop('label' , axis = 1)
y = df_train['label']

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42 , stratify=y)

y_train.value_counts()
y_test.value_counts()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

X.columns

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
sqaure_feature = ['acc_r' , 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_feature = [f for f in df_train.columns if '_temp_' in f]
freq_feature = [f for f in df_train.columns if ('_freq' in f) or ('_pse' in f)]
cluster_feature = ['cluster']

len(time_feature)
len(freq_feature)

fearure_set_1 = basic_features
fearure_set_2 = list(set(basic_features + sqaure_feature + pca_features)) # put into set first to remove any duplicates
fearure_set_3 = list(set(fearure_set_2 + time_feature ))
fearure_set_4 = list(set(fearure_set_3 + freq_feature + cluster_feature  ))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 10


selected_features, ordered_features, ordered_scores = learner.forward_selection( max_features , X_train ,y_train)

selected_features = ['pca_1',
 'gyr_r_freq_0.0_Hz_ws_14',
 'acc_x_freq_0.0_Hz_ws_14',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_z_temp_mean_ws_5',
 'gyr_y_temp_mean_ws_5',
 'acc_z_freq_1.429_Hz_ws_14',
 'gyr_r_freq_2.5_Hz_ws_14',
 'acc_z_freq_weighted',
 'gyr_y_freq_1.429_Hz_ws_14']

ordered_scores = [0.8876249569114099,
 0.9734574284729404,
 0.98931402964495,
 0.9979317476732161,
 0.9993105825577387,
 0.9993105825577387,
 0.9993105825577387,
 0.9993105825577387,
 0.9993105825577387,
 0.9993105825577387]


plt.figure(figsize=(5,10))
plt.plot(range(1 , len(selected_features)+1) , ordered_scores )



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    fearure_set_1,
    fearure_set_2,
    fearure_set_3,
    fearure_set_4,
    selected_features
]

feature_names = [
    'fearure_set_1',
    'fearure_set_2',
    'fearure_set_3',
    'fearure_set_4',
    'selected_features'
]

iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])



# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df = score_df.sort_values(by = 'accuracy' , ascending=False)

plt.figure(figsize=(10,10))
sns.barplot(data= score_df , x = 'model' , y= 'accuracy' , hue='feature_set')
plt.legend(loc='lowe right')
# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(class_train_y, class_test_y, class_train_prob_y, class_test_prob_y,) = learner.random_forest( X_train[fearure_set_4], y_train, X_test[fearure_set_4], gridsearch=True)

accuracy_score(y_test ,class_test_y) 

classes =  class_test_prob_y.columns

cm = confusion_matrix(y_test ,class_test_y ,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
participant_df =  df.drop(['set' , 'category'] ,axis = 1)

X_train =  participant_df[participant_df['participant'] != 'A'].drop('label' , axis = 1)
y_train =  participant_df[participant_df['participant'] != 'A']['label']

X_test =  participant_df[participant_df['participant'] == 'A'].drop('label' , axis = 1)
y_test =  participant_df[participant_df['participant'] == 'A']['label']


X_train= X_train.drop('participant' ,axis = 1)
X_test = X_test.drop('participant' ,axis = 1)

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


(class_train_y, class_test_y, class_train_prob_y, class_test_prob_y,) = learner.random_forest( X_train[fearure_set_4], y_train, X_test[fearure_set_4], gridsearch=True)

accuracy_score(y_test ,class_test_y) 

classes =  class_test_prob_y.columns

cm = confusion_matrix(y_test ,class_test_y ,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------00000000000

(class_train_y, class_test_y, class_train_prob_y, class_test_prob_y,) = learner.feedforward_neural_network( X_train[selected_features], y_train, X_test[selected_features], gridsearch=False)

accuracy_score(y_test ,class_test_y) 

classes =  class_test_prob_y.columns

cm = confusion_matrix(y_test ,class_test_y ,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()