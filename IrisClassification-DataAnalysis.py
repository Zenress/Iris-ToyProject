from sklearn import preprocessing
import pandas as pd
import pandas_profiling as pp

# Handling the Data and Analysing it
# 
#Making a label encoder variable because the last column of the Iris Dataset is of the string datatype
label_encoder = preprocessing.LabelEncoder()

#Making column names for the dataset so it's easier to seperate different parts of the dataset
column_names = ["sepal length","sepal width","petal length","petal width","class"]
#Reading the dataset with no headers and the column names i made above as the column names
iris_df = pd.read_csv("irisdata.csv", header=None, names=column_names)
print(iris_df)


#Number encoding the class (last column) so that it's a numerical representation of the 3 classes.
iris_class_names = iris_df["class"]
iris_df["class"] = label_encoder.fit_transform(iris_df["class"])
print(iris_df)


#For exploratory Analysis of the data
iris_profile = pp.ProfileReport(iris_df, title="Iris Data Profile Report", explorative=True)
iris_profile.to_file("irisreport.html")