# Iris ToyProject
This project is a Classification model using the Iris Dataset. The model used is a DecisionTree and it is made primarily using SKlearn


This project is meant to serve as a sort of ability guaging to see what i can do without googling the solution like crazy.
The project also helped me with building a more team oriented mindset and understanding of how i should structure repositories, the code and this read me file. So that other people might be able to understand it easier

I've learned alot about Data analysis, Github repositories, keeping library imports down and trying to do things using more common libraries.

The libraries used are:
Pandas
Pandas Profiling
SKlearn (StratifiedKFold, Preprocessing, Metrics and DecisionTreeClassifier)
Matplotlib Pyplot
Pickle
Numpy

You can remove Matplotlib and SKLearn.Metrics if you only want to run a prediction on the already trained model

#Information about the files
- The files are split into a group of 4 distinctive files all meant to serve a different purpose in the overall project. 
- There is the DataAnalysis File which serves to show you information, correlations, maximum and minimum, distributions and so on via a profiling module. 
There is also plenty to understand from reading the Console/Terminal printout that the file gives you
- There is the Prediction file which is meant to predict on the saved training model from the Training file, it has a specific format you should write your prediction in,
a maximum and minimum number as well as a datatype, you use it by following the instructions given in the Console/Terminal
- Then there is the training files, there are 2. One of them is completely void of graphs and is mostly supposed to be used when you just want to retrain the model, edit a bit of the parameters
and things like that. The training file that has +graphs in the name is the file that prints out graphs and information about how the KFolded Dataset functions with the DecisionTreeClassifier


How do i retrain the model?
A: You uncomment the code from Line 66-102. This will make it retrain the model and then save. There is also some graphs that are run so you can see that the model is behaving

I can't predict anything?
A: You are probably not writing the text in the correct format. It needs to be comma seperated and it also only needs to be 4 numbers. Any more or less and it will not accept it
