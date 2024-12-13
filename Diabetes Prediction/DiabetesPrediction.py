#Importing Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis

diabetes_dataset = pd.read_csv("datasets/diabetes.csv")
print(diabetes_dataset.head(10))
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset["Outcome"].value_counts())
print(diabetes_dataset.groupby("Outcome").mean())


#seperating the data and labels

X  = diabetes_dataset.drop(columns="Outcome",axis=1)
Y = diabetes_dataset["Outcome"]

#Data Standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)


X = standardized_data
Y = diabetes_dataset["Outcome"]


#Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)


#Training the model

classifier =svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)


#Model Evaluation

#accuracy score on training data

X_train_prediction = classifier.predict(X_train)
training_data_Accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_Accuracy)

#accuracy score on test data

X_test_prediction = classifier.predict(X_test)
testing_data_Accuracy = accuracy_score(X_test_prediction, Y_test)
print(testing_data_Accuracy)

#Making a predictive

input_data =(4,110,92,0,0,37.6,0.191,30)
nparr = np.asarray(input_data)
input_data_reshaped = nparr.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if(prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")




