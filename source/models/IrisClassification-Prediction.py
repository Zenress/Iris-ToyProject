import pickle
import numpy as np
import yaml
#Using configuration file for variables
with open("configuration/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Class names for representing the string version of the encoded ints
class_names = [cfg["class_names"]["class_nr1"],cfg["class_names"]["class_nr2"],cfg["class_names"]["class_nr3"]]
print(class_names)

loaded_model = pickle.load(open(cfg["file_paths"]["model_path"], 'rb'))


def prediction(): 
    """_summary_
    Conditions checked for:
    Datatype: Float
    Length: 4 floats seperated by ,
    Maximum: 10
    Minimum 0.1
    Along with user instructions to follow

    Try:
     Checks for different input requirements
    Raises:
        ValueError: _description_
    """
    print("Write the data you want to be predicted:")
    print("Input Format: Sepal Length, Sepal Width, Petal Length and Petal Width")
    print("Datatype: Float")
    print("Highest Possible Number: 10")
    print("Lowest Possible Number: 0.1")
    unedited_input = input("Write here: ")
    edited_input = unedited_input.split(',')
    
    i=0 
    try:
        for number in edited_input:
            if float(number) >= cfg["input_min_values"][i] and float(number) <= cfg["input_max_values"][i]:  
                i+=1
                continue
            else:
                raise ValueError()
        
        if len(edited_input) == 4:
            class_value = loaded_model.predict(np.reshape(edited_input,(1,4)))
            print(class_value,'=',class_names[class_value[0]])
            
        else:
            print("You have not fulfilled the format requirements")
            print("Try Again \n \n \n")
            prediction()
            
    except ValueError:
        print("Please enter a Float that's between 0.1 and 11 \n \n \n ")
        prediction()

prediction()

#Needs an anaconda activate command explanation
#Anaconda prompt might not be necessary, just a suggestion
#conda environment (link to anaconda) & in sources
#Check what is necessary for someone who knows their way around a terminal
#Read through the readme so it's more relevant and less info dumping when unnecessary
#Make arrays actually be arrays in yaml
#Make as many variables redundant
#Training split should be more flexible so it isn't a static feature selection (features  list for the x features)(label list for labels(class))
#Translation map? Storing the encoder results to? How to go back after encoding something
# profile reports don't need any shuffling
# Configuration for every variable name?
#input needs to disallow non-unicode if i have time