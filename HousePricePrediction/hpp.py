import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
import matplotlib
matplotlib.use("TkAgg")


house_price_dataset = fetch_california_housing()

data = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
data["price"]=house_price_dataset.target

print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.isnull().describe())

#Understanding the correlation between various features in the dataset

#1. Positive Correlation
#2. Negative Correlation


corr = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar=True,square=True,fmt="1f",annot_kws={"size":8},cmap="Reds")
plt.show()

#Splitting the data and target

X = data.drop(columns="price", axis=1)
Y = data["price"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

#XGBoost Regressor

model = XGBRegressor()
model.fit(X_train,Y_train)

#train data prediction
train_data_pred = model.predict(X_train)
score1 = metrics.r2_score(Y_train, train_data_pred)
score2 = metrics.mean_absolute_error(Y_train, train_data_pred)

print("R squared error: ", score1)
print("Mean Absolute Error: ", score2)

#visualizing the actual prices and predicted prices
plt.scatter(Y_train, train_data_pred)
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("Predicted vs True Value")
plt.show()

#test data prediction
test_data_pred = model.predict(X_test)
score3 = metrics.r2_score(Y_test, test_data_pred)
score4 = metrics.mean_absolute_error(Y_test, test_data_pred)

print("R squared error: ", score3)
print("Mean Absolute Error: ", score4)

#visualizing the actual prices and predicted prices
plt.scatter(Y_test, test_data_pred)
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("Predicted vs True Value")
plt.show()