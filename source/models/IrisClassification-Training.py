#Imports used in the Project
import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = 'models/'#TODO: Create more like this
DATASET_PATH = 'source/data/'
CONFIG_PATH = 'configuration/config.yaml'

#Using configuration file for variables
with open(CONFIG_PATH, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument(cfg['cmd_arguments'], help="Enables Graphing", action="store_true")

#Assigning custom column headers while reading the csv file
dataset_df = pd.read_csv(DATASET_PATH+cfg["dataset_name"], header=None, names=cfg["column_names"])

#Labels based on the single categorical column in the dataset

label_encoder = preprocessing.LabelEncoder()

#Encoding the categorical column header to an int datatype
dataset_df[cfg["label_name"]] = label_encoder.fit_transform(dataset_df[cfg["label_name"]])

X = dataset_df[cfg["features"]]
y = dataset_df[cfg["label_name"]]  

dtc_model = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])
data_kfolded = StratifiedKFold(n_splits=cfg["kfold_settings"]["nr_splits"], 
                                shuffle=cfg["kfold_settings"]["shuffle"], 
                                random_state=cfg["kfold_settings"]["random_state"])  # Randomstate for uniform results
                
def train_model(dtc_model,data_kfolded):
    """_summary_
    Train Model Function that trains a DecisionTreeClassifier using a KFolded dataset
    It splits the dataset between x and y training and testing variables
    
    Args:
        dtc_model (DecisionTreeClassifier): An untrained DecisionTreeClassifier used for classifying the KFolded Iris Dataset
        data_kfolded (StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in data_kfolded.split(X,y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        dtc_model = dtc_model.fit(X_train,y_train)
        print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test, dtc_model.predict(X_test))}, doublecheck: {dtc_model.score(X_test,y_test)}")
    
        i += 1

def graphing(): #TODO: Remove Dependency on another function
    """_summary_
    Plots and graphs the dataset and training progress
    """
    occurance_df = []
    round = 1
    for train_index, test_index in data_kfolded.split(X,y):
        o_train = y.iloc[train_index].value_counts()
        o_train.name = f"train {round}"
        o_test = y.iloc[test_index].value_counts()
        o_test.name = f"test {round}"
        df = pd.concat([o_train, o_test], axis=1, sort=False)
        df["|"] = "|"
        occurance_df.append(df)
        
        plt.scatter(x=y.iloc[train_index].index,y=y.iloc[train_index],label="train")
        plt.scatter(x=y.iloc[test_index].index,y=y.iloc[test_index],label="test")
        plt.legend()
        plt.show()
        round+=1
        
    print(pd.concat(occurance_df,axis=1, sort= False))


train_model(dtc_model,data_kfolded)  #TODO Run graphing outside of Train model?

args = parser.parse_args()
if args.graphs:
    graphing() #TODO: Don't run unless needed

dtc_model_and_encoder_mapping = {
    "model": dtc_model,
    "encoder_mappings": label_encoder.classes_,
}

#Saves the file to the given path
pickle.dump(dtc_model_and_encoder_mapping, open(MODEL_PATH+cfg["model_and_encoder_name"], 'wb'))