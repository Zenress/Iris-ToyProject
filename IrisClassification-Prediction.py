import pickle
import numpy as np


#Creating the class names in a seperate list so it can be used to tell the user which Iris it predicted
class_names = ['Iris Setosa','Iris Versicolor','Iris Virginica']

#The name of the modelfile, if you change the filename in the training file you'll have to update it here as well
#The model is loaded using pickle and the filename that you have given
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
    
    #Try except statement that makes sure what you write is a Float
    try:
        #Checking if the value is in the range allowed
        for each in edited_input:
            x = float(each)
            if float(each) > 0.1 and float(each) < 11:
                continue
            else:
                raise ValueError()            
        
        #Checking if you have entered 4 sperate float numbers
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