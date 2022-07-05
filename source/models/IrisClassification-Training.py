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


label_encoder = preprocessing.LabelEncoder()

#Column_names made to assign as headers of the dataset
column_names = [cfg["column_names"]["column_nr1"],cfg["column_names"]["column_nr2"]
                ,cfg["column_names"]["column_nr3"],cfg["column_names"]["column_nr4"]
                ,cfg["column_names"]["column_nr5"]]

#Assigning custom column headers while reading the csv file
iris_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=column_names)

#Encoding the last column header to an int datatype
iris_class_names = iris_df["class"]
iris_df[cfg["column_names"]["column_nr5"]] = label_encoder.fit_transform(iris_df[cfg["column_names"]["column_nr5"]])




dtc = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])
iris_kfold_n5 = StratifiedKFold(n_splits=cfg["kfold_settings"]["nr_splits"], 
                                shuffle=cfg["kfold_settings"]["shuffle"], 
                                random_state=cfg["kfold_settings"]["random_state"])  # Randomstate for uniform results

occurance_df = []

def train_model(dtc,iris_kfold_n5):
    """_summary_
    Train Model Function that trains a DecisionTreeClassifier using a KFolded dataset
    It splits the dataset between x and y training and testing variables
    
    Args:
        dtc (DecisionTreeClassifier): An untrained DecisionTreeClassifier used for classifying the KFolded Iris Dataset
        iris_kfold_n5 (StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in iris_kfold_n5.split(iris_df,iris_df[cfg["column_names"]["column_nr5"]]):
        x_train = iris_df.iloc[train_index].loc[:, column_names[:4]]
        x_test = iris_df.iloc[test_index].loc[:, column_names[:4]]
        y_train_labels = iris_df.iloc[train_index].loc[:, cfg["column_names"]["column_nr5"]]
        y_test_labels = iris_df.loc[test_index].loc[:, cfg["column_names"]["column_nr5"]]

        dtc = dtc.fit(x_train,y_train_labels)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")

        i += 1
        
train_model(dtc,iris_kfold_n5)

#Saves the file to the given path
pickle.dump(dtc,open(cfg["file_paths"]["model_path"], 'wb'))