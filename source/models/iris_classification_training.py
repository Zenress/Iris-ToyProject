"""
Training file used for training the DecisionTreeClassifier.
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


def arguments_handler():
    """
    Handle commandline arguments

    Handles arguments that are passed through during the run command.
    Available arguments are: --graphs

    Returns:
        Namespace: Arguments that were passed through command line

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs", #Does not allow explicit paremeter name=
        help="Enables Graphing",
        action="store_true"
        )
    args = parser.parse_args()
    return args

def read_dataset_and_encode(label_name, dataset_name, column_names, features):
    label_encoder = preprocessing.LabelEncoder()

    #Assigning custom column headers while reading the csv file
    dataset_df = pd.read_csv(DATASET_PATH+dataset_name,
                             header=None,
                             names=column_names)

    #Encoding the categorical column header to an int datatype
    dataset_df[label_name] = label_encoder.fit_transform(dataset_df[label_name])

    X = dataset_df[list(features)]
    y = dataset_df[label_name]

    return X, y, label_encoder

def train_model(dtc_model, train_index, test_index, X, y):
    """
    Train DecisionTreeClassifier Model using Iris Dataset.

    Trains a DecisionTreeClassifier using a KFolded dataset.
    Splits dataset between feature columns and labels,
    and then further splits them into a training and testing set.

    Args:
        dtc_model (sklearn.tree.DecisionTreeClassifier): An untrained DecisionTreeClassifier,
            used for classifying the KFolded Iris Dataset
        data_kfolded (sklearn.model_selection.StratifiedKFold): A KFolded Dataset
        X (pandas.DataFrame) The Feature columns of the dataset
        y (pandas.PandasArray) The Label column of the dataset

    """
    i = 1
    X_train = X.iloc[train_index].values
    X_test = X.iloc[test_index].values
    y_train = y.iloc[train_index].values
    y_test = y.iloc[test_index].values

    dtc_model = dtc_model.fit(X_train,y_train)
    print((f"Accuracy for fold nr. {i} on test set:"
            f" {metrics.accuracy_score(y_test, dtc_model.predict(X_test))} - "
            f"Double check: {dtc_model.score(X_test,y_test)}"))

    i += 1

def graphing(train_index, test_index, occurance_df, round_nr, y):
    """
    Plot dataset and training progress to graph.

    Uses Matplotlib to plot a graph with how the dataset was distributed
    in the different KFold splits.
    Afterwards it tells you how it distributed the label records throughout the splits.
    
    Args:
        train_index (_type_): _description_
        test_index (_type_): _description_
        occurance_df (_type_): _description_
        round_nr (_type_): _description_
        y (_type_): _description_

    Returns:
        pandas.DataFrame: _description_
    """
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

    return occurance_df

def save_file(model, encoder, model_and_encoder_name):
    """
    Save model and encoder mappings.

    Saving the trained model and encoder mappings that was used in the model.
    A dictionary is used to save both objects and pickle them into a single file.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): trained model from unpickled dictionary
        encoder (sklearn.preprocessing.LabelEncoder): labelencoder from unpickled dictionary
        model_and_encoder_name (str) string derived from the dictionary called cfg

    """
    dtc_model_and_encoder_mapping = {
        "model": model,
        "encoder_mappings": encoder.classes_,
    }

    pickle.dump(dtc_model_and_encoder_mapping, open(MODEL_PATH+model_and_encoder_name, 'wb'))

def main():
    """
    Execute at code runtime.

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
    args = arguments_handler()

    with open(CONFIG_PATH, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    X, y, label_encoder = read_dataset_and_encode(
        label_name=cfg["label_name"],
        dataset_name=cfg["dataset_name"],
        column_names=cfg["column_names"],
        features=cfg["features"]
        )

    dtc_model = DecisionTreeClassifier(criterion=cfg["decisiontree_settings"]["criterion"])

    indices_kfold = StratifiedKFold(
        n_splits=cfg["kfold_settings"]["nr_splits"],
        shuffle=cfg["kfold_settings"]["shuffle"],
        random_state=cfg["kfold_settings"]["random_state"]
        )
        # Randomstate for uniform results

    round_nr = 1
    for train_index, test_index in indices_kfold.split(X,y):
        train_model(
            dtc_model=dtc_model,
            train_index=train_index,
            test_index=test_index,
            X=X,
            y=y
            )

        if args.graphs is True:
            occurance_df = []
            occurances_df = graphing(
                train_index=train_index,
                test_index=test_index,
                occurance_df=occurance_df,
                round_nr=round_nr,
                y=y
                )
            round_nr+=1
            print(pd.concat(occurances_df,axis=1, sort= False))

    save_file(
        model=dtc_model,
        encoder=label_encoder,
        model_and_encoder_name=cfg["model_and_encoder_name"]
        )

if __name__ == "__main__":
    main()
    