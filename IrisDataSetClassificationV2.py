#Switching CPU operation instructions to AVX AVX2
import os

from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

# 
# Handling the necessary imports
# 
#Imports used in the Project
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import numpy as np
# import pandas_profiling as pp


# Handling the Data and Analysing it
# 
#Making a label encoder variable because the last column of the Iris Dataset is of the string datatype
label_encoder = preprocessing.LabelEncoder()

#Making column names for the dataset so it's easier to seperate different parts of the dataset
column_names = ["sepal length","sepal width","petal length","petal width","class"]
possible_class_names = ['Iris Setosa','Iris Versicolor','Iris Virginica']
#Reading the dataset with no headers and the column names i made above as the column names
iris_df = pd.read_csv("irisdata.csv", header=None, names=column_names)
print(iris_df)


#Number encoding the class (last column) so that it's a numerical representation of the 3 classes.
iris_class_names = iris_df["class"]
iris_df["class"] = label_encoder.fit_transform(iris_df["class"])
print(iris_df)


#For exploratory Analysis of the data
# iris_profile = pp.ProfileReport(iris_df, title="Iris Data Profile Report", explorative=True)
# iris_profile.to_file("irisreport.html")


# Handling the Model
# 
# Initiating the Model and creating a KFold
dtc = DecisionTreeClassifier(criterion="entropy")
iris_kfold_n5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

#Code for the exploratory Data Analysis of the now Kfolded Iris Dataset
# kf5report_df = pd.DataFrame(data=iris_df)
# kf5report_df = shuffle(kf5report_df)
# kf5report_df.reset_index(inplace=True, drop=True)
# iris_kfold_profile = pp.ProfileReport(kf5report_df, title="Iris KFolded Data Report", explorative=True)
# iris_kfold_profile.to_file("iriskfoldreport.html")

occurance_df = []

# Using a for loop to run and train the model through all the folds created by kf3
# i = 1
# for train_index, test_index in iris_kfold_n5.split(iris_df,iris_df["class"]):
#     #Splitting the dataset
#     x_train = iris_df.iloc[train_index].loc[:, column_names[:4]]
#     x_test = iris_df.iloc[test_index].loc[:, column_names[:4]]
#     y_train_labels = iris_df.iloc[train_index].loc[:, "class"]
#     y_test_labels = iris_df.loc[test_index].loc[:, "class"]
    
#     #Training and printing the result of each iteration
#     dtc = dtc.fit(x_train,y_train_labels)
#     print(f"Accuracy for the fold nr. {i} on the test set: {metrics.accuracy_score(y_test_labels, dtc.predict(x_test))}, doublecheck: {dtc.score(x_test,y_test_labels)}")
    
#     #Counting occurances
#     o_train = iris_class_names[train_index].value_counts()
#     o_train.name = f"train {i}"
#     o_test = iris_class_names[test_index].value_counts()
#     o_test.name = f"test {i}"
#     df = pd.concat([o_train, o_test], axis=1, sort=False)
#     df["|"] = "|"
#     occurance_df.append(df)
    
#     i += 1


#     # Plotting the data 
#     #
#     #Using the data and indexes derived from the Kfold with the class names Dataframe together to plot out a graph
#     plt.scatter(x=y_train_labels.index,y=iris_class_names[train_index],label="train")
#     plt.scatter(x=y_test_labels.index,y=iris_class_names[test_index],label="test")
#     plt.legend()
#     plt.show()

# #Here is the number of occurances
# print(pd.concat(occurance_df,axis=1, sort=False))

filename = 'irisdata_classification_model.sav'
# pickle.dump(dtc,open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


#predict?
unedited_input = input("Write the data you want to be predicted in the following format seperated by comma: Sepal Length, Sepal Width, Petal Length and Petal Width")
edited_input = unedited_input.split(',')
print(len(edited_input))
if len(edited_input) == 4:
    print(unedited_input)
    for each in edited_input:
        print(each)
    class_value = loaded_model.predict(np.reshape(edited_input,(1,4)))
    print(class_value,' = ',possible_class_names[class_value[0]])
else:
    print("You have not fulfilled the format requirements")
#readme