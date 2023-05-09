import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print('Overall accuracy: ', acc)
for i in range(len(x_test)):
    print("Prediction: ", classes[y_pred[i]], "Data: ", x_test[i], "Actual: ", classes[y_test[i]])