import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

# summarise observations by class label
# WHAT DOES COUNTER DO?
counter = Counter(data['Species'])

# logistic regression for multi-class classification using built-in one-vs-rest

# define the dataset
x = data.drop(['Species', 'PetalLengthCm', 'PetalWidthCm'], axis=1)
y = data['Species']
for label in y:
	if label == 'Iris-setosa':
		label = 0
	if label == 'Iris-versicolor':
		label = 1
	if label == 'Iris-virginica':
		label = 2


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1, shuffle=True)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# define the model
model = LogisticRegression(multi_class='ovr', max_iter = 1000)

# fit model
model.fit(x_train, y_train)

yhat =  model.predict(x_test)
accuracy = accuracy_score(y_test, yhat)

print(classification_report(y_test, yhat))
print('Accuracy: %.3f' % accuracy)