import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
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

# summarise observations by class label
# WHAT DOES COUNTER DO?
counter = Counter(data['Species'])

# define the dataset
x = data.drop(['Species', 'PetalLengthCm', 'PetalWidthCm'], axis=1)
y = data.Species

# decision tree for multi-class classification

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# define the model
# HOW DOES THIS WORK INTERNALLY?
model = DecisionTreeClassifier(max_depth=2)

# fit the model
model.fit(x_train, y_train)

yhat = model.predict(x_test)

# create a confusion matrix
cm = confusion_matrix(y_test, yhat)
print(cm)

accuracy = accuracy_score(y_test, yhat)

print(classification_report(y_test, yhat))
print('Accuracy: %.3f' % accuracy)