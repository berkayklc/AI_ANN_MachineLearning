# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:23:46 2022

@author: berkay
"""

import pandas as pd
import numpy as np





dataset=pd.read_csv(r"C:\Users\berka\AirlanePassenger.csv")

X=dataset.iloc[:,2:26].values
y=dataset.iloc[:, 24].values

#Encoding Categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_x_0=LabelEncoder()
X[:, 0]=labelencoder_x_0.fit_transform(X[:, 0])
labelencoder_x_1=LabelEncoder()
X[:, 1]=labelencoder_x_1.fit_transform(X[:, 1])
labelencoder_x_3=LabelEncoder()
X[:, 3]=labelencoder_x_3.fit_transform(X[:, 3])
labelencoder_x_4=LabelEncoder()
X[:, 4]=labelencoder_x_4.fit_transform(X[:, 4])
onehotencoder=OneHotEncoder()
X=onehotencoder.fit_transform(X).toarray()
X=X[:, 0:]




#Training the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,train_size=0.9,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Create Neural Network and Compile Hidden Layer
from sklearn.neural_network import MLPClassifier
Mlp_classifier=MLPClassifier(hidden_layer_sizes=(10,10,10,10),max_iter=10000)
Mlp_classifier.fit(X_train, y_train.ravel())




#Make Predictions on Test Value
predictions=Mlp_classifier.predict(X_test)





#Scale Predictions and Performance of Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))








