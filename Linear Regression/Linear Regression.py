import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

'''
We should choose to use Linear Regression whenever you are dealt with data points that are
correlated in some way. For example, the student grades that are used in this project
are correlated, because using grade 1 and grade 2 you have a good chance of predicting grade 3. 

Picturing a graph with data points that are closely tied together you could draw a line in what
you would think as the "middle-point" of the graph, with respect to the starting values on the x 
and y axis.  If the data points are close together you can predict where the line will start and
where the line will end on the graph. This is Linear Regression - the "best-fit" line.
'''

data = pd.read_csv('student-mat.csv', sep=';')

# What this does is narrow (trim) down the data to the values that are relevant,
# in this case the grades, the study time, failures, and the absences
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# label
predict = "G3"

# This will drop the data that corresponds to the prediction data ("G3")
X = np.array(data.drop(labels=[predict], axis=1))
# We only care about the G3 data for the Y
Y = np.array(data[predict])

'''
-- Training --
1. x_train is the subset of data the model will assess - more specifically the X (lack of G3) values
2. y_train is the target values (or labels) in x_train (G3 label)
-- Evaluation --
3. x_test is the subset of data, respective to X, that is essentially the correct answer
4. y_test is the actual target value of x_test that will be used to complete the evaluation
'''
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

# Now we apply the linear regression to find the best fit line
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# We test the train values with the test values to find the accuracy
accuracy = linear.score(x_test, y_test)

# Since linear regression finds the best fit line we can actually just get those (y=mx+b) values.
# I'm keeping it simple with y=mx+b because when we're going beyond 2D space (which we are),
# the formula can get tricky
print('Total accuracy: ',accuracy)
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# Now that we have our accuracy and data, we can predict a students grade
predictions = linear.predict(x_test)

print()
for x in range(len(predictions)):
    print('Prediction: ', predictions[x], 'Data: ',x_test[x], 'Target: ', y_test[x])