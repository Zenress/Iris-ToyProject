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

#Assigning custom column headers while reading the csv file
iris_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=cfg["column_names"])

#Encoding the last column header to an int datatype
iris_df["class"] = label_encoder.fit_transform(iris_df["class"])
iris_features = iris_df.drop(columns="class")

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
    for train_index, test_index in iris_kfold_n5.split(iris_df,iris_df["class"]):
        x_train = iris_features.iloc[train_index]
        x_test = iris_features.iloc[test_index]
        y_train_labels = iris_df["class"].iloc[train_index]
        y_test_labels = iris_df["class"].iloc[test_index]

        dtc = dtc.fit(x_train,y_train_labels)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")

        i += 1
        
train_model(dtc,iris_kfold_n5)

#Saves the file to the given path
pickle.dump(dtc,open(cfg["file_paths"]["model_path"], 'wb'))

#Saves encoder mapping to a pkl file
encoder_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
pickle.dump(encoder_mapping,open(cfg["file_paths"]["encoder_mappings"],"wb"))