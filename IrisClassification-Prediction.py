from multiprocessing.sharedctypes import Value
import pickle
import numpy as np
import os

from sympy import Predicate

class_names = ['Iris Setosa','Iris Versicolor','Iris Virginica']
filename = 'irisdata_classification_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

# Predicting the class of the features you write. 
def predection(): 
    print("Write the data you want to be predicted:")
    print("Input Format: Sepal Length, Sepal Width, Petal Length and Petal Width")
    print("Datatype: Float")
    print("Highest Possible Number: 10")
    print("Lowest Possible Number: 0.1")
    unedited_input = input("Write here: ")
    edited_input = unedited_input.split(',')
    try:
        for each in edited_input:
            x = float(each)
            if float(each) > 0.1 and float(each) < 11:
                continue
            else:
                raise ValueError() 
                
            
        if len(edited_input) == 4:
            class_value = loaded_model.predict(np.reshape(edited_input,(1,4)))
            print(class_value,'=',class_names[class_value[0]])
        else:
            print("You have not fulfilled the format requirements")
            print("Try Again \n \n \n")
            predection()
    except ValueError:
        print("Please enter a Float that's between 0.1 and 11 \n \n \n ")
        predection()
    

predection()