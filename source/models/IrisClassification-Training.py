#Imports used in the Project
import pickle
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Assigning custom column headers while reading the csv file
dataset_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=cfg["column_names"])

#Labels based on the single categorical column in the dataset
labels = dataset_df.select_dtypes(include=['object'])

label_encoder = preprocessing.LabelEncoder()

#Encoding the categorical column header to an int datatype
dataset_df[str(labels.columns.values[0])] = label_encoder.fit_transform(dataset_df[str(labels.columns.values[0])])
dataset_features = dataset_df.drop(columns=labels)

dtc = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])
dataset_kfolded = StratifiedKFold(n_splits=cfg["kfold_settings"]["nr_splits"], 
                                shuffle=cfg["kfold_settings"]["shuffle"], 
                                random_state=cfg["kfold_settings"]["random_state"])  # Randomstate for uniform results

occurance_df = []

def train_model(dtc,dataset_kfolded):
    """_summary_
    Train Model Function that trains a DecisionTreeClassifier using a KFolded dataset
    It splits the dataset between x and y training and testing variables
    
    Args:
        dtc (DecisionTreeClassifier): An untrained DecisionTreeClassifier used for classifying the KFolded Iris Dataset
        dataset_kfolded (StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in dataset_kfolded.split(dataset_df,dataset_df[str(labels.columns.values[0])]):
        x_train = dataset_features.iloc[train_index]
        x_test = dataset_features.iloc[test_index]
        y_train_labels = dataset_df[str(labels.columns.values[0])].iloc[train_index]
        y_test_labels = dataset_df[str(labels.columns.values[0])].iloc[test_index]

        dtc = dtc.fit(x_train,y_train_labels)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")

        i += 1
        
train_model(dtc,dataset_kfolded)

#Saves the file to the given path
pickle.dump(dtc,open(cfg["file_paths"]["model_path"], 'wb'))

#Saves encoder mapping to a pkl file
encoder_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
pickle.dump(encoder_mapping,open(cfg["file_paths"]["encoder_mappings"],"wb"))