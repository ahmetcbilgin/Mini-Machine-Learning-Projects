import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


loan_dataset = pd.read_csv("datasets/loan.csv")

print(loan_dataset.head())
print(loan_dataset.shape)
print(loan_dataset.describe())
print(loan_dataset.isnull().sum())

loan_dataset = loan_dataset.dropna()

print(loan_dataset.isnull().sum())

#label encoding
loan_dataset.replace({"Loan_Status" : {"N":0, "Y":1}}, inplace=True)

#replacing the value of 3+ to 4
print(loan_dataset["Dependents"].value_counts())
loan_dataset = loan_dataset.replace(to_replace="3+", value=4)
print(loan_dataset["Dependents"].value_counts())

#Data Visualization

#education & loan status
sns.countplot(x="Education",hue="Loan_Status",data=loan_dataset)
plt.show()

#marital status & loan status
sns.countplot(x="Married",hue="Loan_Status",data=loan_dataset)
plt.show()

#convert categorical columns to numerical values
loan_dataset.replace({"Married" : {"No":0, "Yes":1},"Gender" : {"Female":0, "Male":1},"Self_Employed" : {"No":0, "Yes":1},
                     "Property_Area" : {"Rural":0, "Semiurban":1, "Urban":2},"Education" : {"Not Graduate":0, "Graduate":1}},inplace=True)

print(loan_dataset)
print(loan_dataset.info())

#seperating the data and label
X=loan_dataset.drop(columns=["Loan_ID","Loan_Status"],axis=1)
Y=loan_dataset["Loan_Status"]

#train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)


#training the model

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#Model Evaluation

#accuracy score on training data
X_train_pred = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)

print("Accuracy Score on Training Data : ", training_data_accuracy)

#accuracy score on training data
X_test_pred = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)

print("Accuracy Score on Test Data : ", test_data_accuracy)
