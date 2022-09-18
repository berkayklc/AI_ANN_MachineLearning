# AI_ANN_MachineLearning

1.	INTRODUCTION
           In this project, we conducted a satisfaction analysis of customers traveling by airlane. We used the data source created with the information collected such as gender, customer type, age, flight status, travel reasons. In this data source, apart from customer information, there are satisfaction scores collected under many topics such as food and drink satisfaction, seat comfort satisfaction, luggage comfort, cleaning satisfaction. With all this data set resource, we have created an artificial neural network. We have tried to analyze which type of customer is satisfied and dissatisfied with which reasons with this created network. With this and similar studies, situations of dissatisfaction of airlane travelling will determine previously and take some solutions for these issues previously. 

2.	MATERIALS AND METHODS

This dataset contains an airline passenger satisfaction survey. This data set has been collected under 23 different subject headings;

  1)	Gender: Gender of the passengers (Female, Male)

  2)	Customer Type: The customer type (Loyal customer, disloyal customer)

  3)	Age: The actual age of the passengers

  4)	Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)

  5)	Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)

  6)	Flight distance: The flight distance of this journey

  7)	Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)

  8)	Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient

  9)	Ease of Online booking: Satisfaction level of online booking

  10)	Gate location: Satisfaction level of Gate location

  11)	Food and drink: Satisfaction level of Food and drink

  12)	Online boarding: Satisfaction level of online boarding

  13)	Seat comfort: Satisfaction level of Seat comfort

  14)	Inflight entertainment: Satisfaction level of inflight entertainment

  15)	On-board service: Satisfaction level of On-board service

  16)	Leg room service: Satisfaction level of Leg room service

  17)	Baggage handling: Satisfaction level of baggage handling

  18)	Check-in service: Satisfaction level of Check-in service

  19)	Inflight service: Satisfaction level of inflight service

  20)	Cleanliness: Satisfaction level of Cleanliness

  21)	Departure Delay in Minutes: Minutes delayed when departure

  22)	Arrival Delay in Minutes: Minutes delayed when Arrival

  23)	Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)
 
 
 -------------------------------CODES--------------------------------------------------------
 
 @author: berkay
"""
#Import pandas and numpy libraries, then define “pd”  for pandas and define “np” for numpy
import pandas as pd 
import numpy as np

#Define path of our dataset document with “read_csv” code of pandas library

dataset=pd.read_csv(r"C:\Users\berka\AirlanePassenger.csv")

#Create X and y variables for parse to our dataset. X variable includes part of information, y variable includes part of result (satisfied – neutral or dissatisfied)

X=dataset.iloc[:,2:26].values
y=dataset.iloc[:, 24].values

#Import “Sklearn Library” to encoding our data for categorical parsing

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Create a variable for labelling one column of objects columns

labelencoder_x_0=LabelEncoder()

#The zeroth index of columns of X dataset include “Female, Male” object. With this labelling process, these objects were transformeded integer format

X[:, 0]=labelencoder_x_0.fit_transform(X[:, 0])

#Create a variable for labelling one column of objects columns

labelencoder_x_1=LabelEncoder()

#The first index of columns of X dataset include “Loyal customer, Disloyal Customer” object. With this labelling process, these objects were transformeded integer format

X[:, 1]=labelencoder_x_1.fit_transform(X[:, 1])

#Create a variable for labelling one column of objects columns

labelencoder_x_3=LabelEncoder()

#The third index of columns of X dataset include “Business Travel, Personal Travel” object. With this labelling process, these objects were transformeded integer format

X[:, 3]=labelencoder_x_3.fit_transform(X[:, 3])

#Create a variable for labelling one column of objects columns

labelencoder_x_4=LabelEncoder()

#The fourth index of columns of X dataset include “Eco, Business” object. With this labelling process, these objects were transformeded integer format

X[:, 4]=labelencoder_x_4.fit_transform(X[:, 4])

#Labelled zeroth index of our dataset array

onehotencoder=OneHotEncoder()
X=onehotencoder.fit_transform(X).toarray()
X=X[:, 0:]

#Import model_selection method of sklearn library for training the dataset into the training set and test set

from sklearn.model_selection import train_test_split
 
 #With use this function for define test and train variables and define percentage of train and test

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=0)

#Using preprocessing method of sklearn library to import StandartScaler for scaling our X_train and X_test variables

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Import neural_network method of sklearn library to import MLPCLassifier for creating neural network

from sklearn.neural_network import MLPClassifier

#Define MLPClassifier and create hidden layer which has 4x10 neurons with size of maximum iteration

Mlp_classifier=MLPClassifier(hidden_layer_sizes=(10,10,10,10),max_iter=10000)

#Fitting to train variables for predicting by MLPClassifier

Mlp_classifier.fit(X_train, y_train.ravel())

#Making a prediction on our test value

predictions=Mlp_classifier.predict(X_test)

#Scaling our predictions and performance of our algorithm by using sklearn library with classification_report method and confusion_matrix method

from sklearn.metrics import classification_report, confusion_matrix

#Printing our conlusions

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

---------------------------------------------------------------------------------------------------------

CONCLUSION

In this study, we have created a network of artificial neurons that can make significantly accurate predictions, such as 99%. Thus, we have learned that artificial neural networks are the most important factor in the field of artificial intelligence. In this study, scikit-learn libraries were used. In addition to this library, keras, tensorflow, Pytorch, etc. libraries can be used for machine learning. 


