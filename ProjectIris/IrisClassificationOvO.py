import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load the dataset

data = pd.read_csv('C:/Users/Nyambura/Documents/Coding/Iris/ProjectIris/Iris.csv')

# Explore the dataset

df_size = data.size
df_shape = data.shape

# print(df_size)	total num of data points
# print(df_shape)	150 rows, 6 columns
# print(data.head())	first five rows of the dataset

# We want to classify by 'Species' (3 distinct values)
# Multi-class classification

counter = Counter(data['Species'])

# define the dataset
x = data.drop(['Species', 'PetalLengthCm', 'PetalWidthCm'], axis=1)
y = data.Species

# SVC for multi-class classification using built-in one-vs-one

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# define the model

# WHAT DOES 'decision_function_shape' DO?
model = SVC(decision_function_shape='ovo')

# fit the model
model.fit(x_train, y_train)

yhat = model.predict(x_test)
accuracy = accuracy_score(y_test, yhat)

print(classification_report(y_test, yhat))
print('Accuracy: %.3f' % accuracy)

