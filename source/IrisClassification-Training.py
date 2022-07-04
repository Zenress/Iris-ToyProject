#Imports used in the Project
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas_profiling as pp
import yaml
with open("./configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Handling the Data
# 
#Making a label encoder variable because the last column of the Iris Dataset is of the string datatype
label_encoder = preprocessing.LabelEncoder()

#Making column names for the dataset so it's easier to seperate different parts of the dataset
column_names = [cfg["column_names"]["column_nr1"],cfg["column_names"]["column_nr2"]
                ,cfg["column_names"]["column_nr3"],cfg["column_names"]["column_nr4"]
                ,cfg["column_names"]["column_nr5"]]
#Reading the dataset with no headers and the column names i made above as the column names
iris_df = pd.read_csv(cfg["file_paths"]["dataset_path"], header=None, names=column_names)

#Number encoding the class (last column) so that it's a numerical representation of the 3 classes.
iris_class_names = iris_df["class"]
iris_df[cfg["column_names"]["column_nr5"]] = label_encoder.fit_transform(iris_df[cfg["column_names"]["column_nr5"]])



# Handling the Model
# 
# Initiating the Model and creating a KFold
dtc = DecisionTreeClassifier(criterion="entropy")
iris_kfold_n5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

occurance_df = []

def train_model(dtc,iris_kfold_n5):
    """_summary_
    Train Model Function that trains a DecisionTreeClassifier using a KFolded dataset
    
    Args:
        dtc (DecisionTreeClassifier): An untrained DecisionTreeClassifier used for classifying the KFolded Iris Dataset
        iris_kfold_n5 (StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in iris_kfold_n5.split(iris_df,iris_df[cfg["column_names"]["column_nr5"]]):
        #Splitting the dataset
        x_train = iris_df.iloc[train_index].loc[:, column_names[:4]]
        x_test = iris_df.iloc[test_index].loc[:, column_names[:4]]
        y_train_labels = iris_df.iloc[train_index].loc[:, cfg["column_names"]["column_nr5"]]
        y_test_labels = iris_df.loc[test_index].loc[:, cfg["column_names"]["column_nr5"]]

        dtc = dtc.fit(x_train,y_train_labels)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")

        i += 1
        
train_model(dtc,iris_kfold_n5)

pickle.dump(dtc,open(cfg["file_paths"]["model_path"], 'wb'))