"""
Training file used for training the DecisionTreeClassifier
"""
import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = 'models/'
DATASET_PATH = 'source/data/'
CONFIG_PATH = 'configuration/config.yaml'


def arguments_handler(cmd_arguments):
    """
    Handle commandline arguments

    Handles arguments that are passed through during the run command.
    Available arguments are: --graphs

    Args:
        cmd_arguments (str): string with the available command line argument
    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(cmd_arguments, help="Enables Graphing", action="store_true")
    args = parser.parse_args()
    return args

def train_model(dtc_model,data_kfolded, X, y):
    """
    Train DecisionTreeClassifier Model using Iris Dataset

    Trains a DecisionTreeClassifier using a KFolded dataset.
    Splits dataset between feature columns and labels,
    and then further splits them into a training and testing set.

    Args:
        dtc_model (sklearn.tree.DecisionTreeClassifier): An untrained DecisionTreeClassifier,
        used for classifying the KFolded Iris Dataset
        data_kfolded (sklearn.model_selection.StratifiedKFold): A KFolded Dataset
    """
    i = 1
    for train_index, test_index in data_kfolded.split(X,y):
        X_train = X.iloc[train_index].values
        X_test = X.iloc[test_index].values
        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values

        dtc_model = dtc_model.fit(X_train,y_train)
        print((f"Accuracy for fold nr. {i} on test set:"
               f" {metrics.accuracy_score(y_test, dtc_model.predict(X_test))} - "
               f"Double check: {dtc_model.score(X_test,y_test)}"))

        i += 1

def graphing(args, data_kfolded, X, y):
    """
    Plot dataset and training progress to graph

    Uses Matplotlib to plot a graph with how the dataset was distributed
    in the different KFold splits.
    Afterwards it tells you how it distributed the label records throughout the splits

    """
    if args.graphs:
        occurance_df = []
        round_nr = 1
        for train_index, test_index in data_kfolded.split(X,y):
            o_train = y.iloc[train_index].value_counts()
            o_train.name = f"train {round_nr}"
            o_test = y.iloc[test_index].value_counts()
            o_test.name = f"test {round_nr}"

            #Concatenate 2 pandas objects along a single dataframe axis
            df = pd.concat([o_train, o_test], axis=1, sort=False)
            df["|"] = "|"
            occurance_df.append(df)

            plt.scatter(x=y.iloc[train_index].index,y=y.iloc[train_index],label="train")
            plt.scatter(x=y.iloc[test_index].index,y=y.iloc[test_index],label="test")
            plt.legend()
            plt.show()
            round_nr+=1

        print(pd.concat(occurance_df,axis=1, sort= False))

def save_file(model,encoder, model_and_encoder_name):
    """
    Save model and encoder mappings

    Saving the trained model and encoder mappings that was used in the model.
    A dictionary is used to save both objects and pickle them into a single file.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): _description_
        encoder (sklearn.preprocessing.LabelEncoder): _description_
    """
    dtc_model_and_encoder_mapping = {
        "model": model,
        "encoder_mappings": encoder.classes_,
    }

    pickle.dump(dtc_model_and_encoder_mapping, open(MODEL_PATH+model_and_encoder_name, 'wb'))

def main():
    """
    Execute at code runtime

    Initiating configuration file ->,
    Labelencoder is initialized -> Dataset is read ->,
    Categorical Label Column is encoded to numerical type ->,
    X is assigned with Features and Y is assigned with Label ->,
    Model and KFold cross validation is initialized and run ->,
    arguments_handler is run to check for cmd line arguments ->,
    the model is trained with train_model ->,
    graphing is run if correct arguments are present ->,
    the finished training and encoding is saved to a file as dictionaries.

    """
    with open(CONFIG_PATH, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    label_encoder = preprocessing.LabelEncoder()

    #Assigning custom column headers while reading the csv file
    dataset_df = pd.read_csv(DATASET_PATH+cfg["dataset_name"],
                             header=None,
                             names=cfg["column_names"])

    #Encoding the categorical column header to an int datatype
    dataset_df[cfg["label_name"]] = label_encoder.fit_transform(dataset_df[cfg["label_name"]])

    X = dataset_df[list(cfg["features"])]
    y = dataset_df[cfg["label_name"]]

    dtc_model = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])

    data_kfolded = StratifiedKFold(n_splits=cfg["kfold_settings"]["nr_splits"],
                                    shuffle=cfg["kfold_settings"]["shuffle"],
                                    random_state=cfg["kfold_settings"]["random_state"])  # Randomstate for uniform results

    args = arguments_handler(cfg['cmd_arguments'])

    train_model(dtc_model,data_kfolded, X, y)

    graphing(args, data_kfolded, X, y)

    save_file(dtc_model,label_encoder,cfg["model_and_encoder_name"])

if __name__ == "__main__":
    main()
    