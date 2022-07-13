#Imports used in the Project
import pickle
import getopt, sys
import matplotlib.pyplot as plt
import numpy as np
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
dataset_df = pd.read_csv("source/data/"+cfg["dataset_name"], header=None, names=cfg["column_names"])

#Labels based on the single categorical column in the dataset

label_encoder = preprocessing.LabelEncoder()

#Encoding the categorical column header to an int datatype
dataset_df[cfg["label_name"]] = label_encoder.fit_transform(dataset_df[cfg["label_name"]])

dtc = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])
dataset_kfolded = StratifiedKFold(n_splits=cfg["kfold_settings"]["nr_splits"], 
                                shuffle=cfg["kfold_settings"]["shuffle"], 
                                random_state=cfg["kfold_settings"]["random_state"])  # Randomstate for uniform results

occurance_df = []
arguments, dummy = getopt.getopt(sys.argv[1:],cfg["cmd_arguments"]["options"],cfg["cmd_arguments"]["long_options"])

X = dataset_df[cfg["features"]]
y = dataset_df[cfg["label_name"]] #TODO: Missing encoder?

def train_model(dtc,dataset_kfolded):
    """_summary_
    Train Model Function that trains a DecisionTreeClassifier using a KFolded dataset
    It splits the dataset between x and y training and testing variables
    
    Args:
        dtc (DecisionTreeClassifier): An untrained DecisionTreeClassifier used for classifying the KFolded Iris Dataset
        dataset_kfolded (StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in dataset_kfolded.split(X,y):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        y_train_labels = y.iloc[train_index]
        y_test_labels = y.iloc[test_index]

        dtc = dtc.fit(x_train,y_train_labels)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")
        
        try:
            for current_arg, dummy in arguments:
                if current_arg in ("-g","--graphs"):
                    o_train = y.iloc[train_index].value_counts()
                    o_train.name = f"train {i}"
                    o_test = y.iloc[test_index].value_counts()
                    o_test.name = f"test {i}"
                    df = pd.concat([o_train, o_test], axis=1, sort=False)
                    df["|"] = "|"
                    occurance_df.append(df)
                    
                    plt.scatter(x=y_train_labels.index,y=y.iloc[train_index],label="train")
                    plt.scatter(x=y_test_labels.index,y=y.iloc[test_index],label="test")
                    plt.legend()
                    plt.show()   
                    
                    if i == cfg["kfold_settings"]["nr_splits"]:
                        print(pd.concat(occurance_df,axis=1, sort= False))
                    
        except getopt.error as err:
            # output error, and return with an error code
            print (str(err))

        i += 1

train_model(dtc,dataset_kfolded)  

#Saves the file to the given path
pickle.dump(dtc,open("models/"+cfg["model_name"], 'wb'))

#Saves encoder mapping to a pkl file
np.save("models/"+cfg["encoder_mappings"],label_encoder.classes_)