import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas_profiling as pp

def unit_test1():
    """
    Test method used to test things
    """
    CONFIG_PATH = 'configuration/config.yaml'

    with open(CONFIG_PATH, "r", encoding='UTF-8') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print(type(cfg))

def main():
    """
    Execute at code runtime
    """
    unit_test1()

if __name__ == "__main__":
    main()
