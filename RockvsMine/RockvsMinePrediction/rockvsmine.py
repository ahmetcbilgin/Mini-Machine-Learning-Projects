
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("datasets/sonar_data.csv",header=None)

print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data.describe())
print(sonar_data[60].value_counts())
print(sonar_data.groupby(60).mean())


#seperating data and labels

X =sonar_data.drop(columns=60,axis=1)
Y= sonar_data[60]

print(X)
print(Y)


#Train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)


#Model training - Logiatic Regression

model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation

#accuracy on training data
x_train_predictions = model.predict(X_train)
training_accuracy = accuracy_score(x_train_predictions, Y_train)
print(training_accuracy)


#accuracy on test data
x_test_predictions = model.predict(X_test)
training_accuracy = accuracy_score(x_test_predictions, Y_test)
print(training_accuracy)
