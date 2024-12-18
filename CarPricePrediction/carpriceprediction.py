import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')


car_dataset = pd.read_csv('datasets/car data.csv')

print(car_dataset.head())
print(car_dataset.shape)
print(car_dataset.describe())
print(car_dataset.info())

print(car_dataset.isnull().sum())

print(car_dataset["Fuel_Type"].value_counts())
print(car_dataset["Seller_Type"].value_counts())
print(car_dataset["Transmission"].value_counts())

car_dataset.replace({"Fuel_Type" : {"Petrol":0, "Diesel":1, "CNG":2},
                     "Seller_Type": {"Dealer":0, "Individual":1},
                     "Transmission":{"Manual":0, "Automatic":1}}, inplace=True)


car_dataset['Selling_Price'].hist()
plt.show()

car_dataset['Car_Name'].unique
car_dataset.drop('Car_Name', axis=1, inplace=True)

X = car_dataset.drop(['Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

#Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

#prediction on training data
training_data_prediciton = lin_reg_model.predict(X_train)

#R squared error
error_score =metrics.r2_score(Y_train, training_data_prediciton)
print(error_score)


plt.scatter(Y_train, training_data_prediciton)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

#prediction on test data
test_data_prediciton = lin_reg_model.predict(X_test)

#R squared error
error_score =metrics.r2_score(Y_test, test_data_prediciton)
print(error_score)

plt.scatter(Y_test, test_data_prediciton)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


#Lasso Regression

lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

#prediction on training data
training_data_prediciton = lass_reg_model.predict(X_train)

#R squared error
error_score =metrics.r2_score(Y_train, training_data_prediciton)
print(error_score)


plt.scatter(Y_train, training_data_prediciton)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

#prediction on test data
test_data_prediciton = lass_reg_model.predict(X_test)

#R squared error
error_score =metrics.r2_score(Y_test, test_data_prediciton)
print(error_score)

plt.scatter(Y_test, test_data_prediciton)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
