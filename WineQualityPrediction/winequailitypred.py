import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


wine_dataset = pd.read_csv('datasets/winequality-red.csv')

print(wine_dataset.head())
print(wine_dataset.isnull().sum())

print(wine_dataset.describe())
sns.catplot(x='quality', data=wine_dataset,kind='count')
plt.show()


corr = wine_dataset.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr, cbar=True, square=True, fmt=".lf", annot_kws ={"size": 8}, cmap="Reds")
plt.show()

X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value : 1 if y_value >= 7 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)

print(f'Test Accuracy: {test_data_accuracy}')

