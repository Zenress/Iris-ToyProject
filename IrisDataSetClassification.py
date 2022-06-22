#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

#Imports used in the Project
import tensorflow as tf
from keras import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Making a label encoder variable because the last column of the Iris Dataset is of the string datatype
label_encoder = preprocessing.LabelEncoder()

#Making column names for the dataset so it's easier to seperate different parts of the dataset
column_names = ["sepal length","sepal width","petal length","petal width","class"]
#Reading the dataset with no headers and the column names i made above as the column names
df = pd.read_csv("Irises ToyProject/irisdata.csv", header=None, names=column_names)
print(df)
#Number encoding the class (last column) so that it's a numerical representation of the 3 classes.
df["class"] = label_encoder.fit_transform(df["class"])
print(df)

#Using scikit learn to split the dataframe into 2 sets using an 80/20 split.
train, test = train_test_split(df, test_size=0.2)

#Popping off the class column only, since that is gonna be used as the labels for each row. We're gonna use it to check wether the model is correct or not in the assumption it made
train_labels = train.pop("class")
test_labels = test.pop("class")
train_x = train
test_x = test

#Making the model using Dense layers with the ReLU Activation Function
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4, )),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])

#Compiling the model using the Adam Optimizer and a SparseCategoricalCrossentropy
model.compile(optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

#Training the model and then afterwards evaluating if it's trained correctly
model.fit(train_x, train_labels, epochs=50)
test_loss, test_acc = model.evaluate(test_x,  test_labels, verbose=2)
