import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

'''
We should choose to use the K-Nearest Neighbors algorithm when dealing with small
to medium sized datasets, non-linear relationships, noisy data, or for quick-prototyping.
For example, the dataset for this project uses cars and predicts their class respective to
the other attributes of the car.

Picturing a graph with grouped values say 1. a group of red dots, 2. a group of green dots, and 3.
a group of blue dots - we can classify a new dot, a black dot, by how close it is to its neighbors
and how close its neighbors are to their neighbors and so on. As humans we can view a 2-D graph of 
the black, red, green, and blue dots and choose which group the black dot belongs to visually. However,
why a computer we have to measure the magnitude between the black dot and its neighbors. KNN takes in a
value k (an odd number) say k = 3, and then takes 3 of closest neighbors surrounding the black dot. If 
there are 2 red dots and 1 green dot neighbors, we can safely assume that the black dot belongs to the
red group of dots because it is closest. This is how KNN works.
'''

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
doors = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

X = list(zip(buying, maint, doors, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(accuracy)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']
for i in range(len(x_test)):
    print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
