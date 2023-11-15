from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

X, y = make_circles(n_samples=1000, factor=0.2, noise=0.4)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
print('\nLogistic regression')
model = LogisticRegression()   #choose the model
model.fit(X_train, y_train)    #teach the model
print(model.score(X_test, y_test))      #print accuracy
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))    #print details

from sklearn.neighbors import KNeighborsClassifier
print('\nKNN')
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from sklearn.tree import DecisionTreeClassifier
print('\nDecission  Tree Clasifier')
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from sklearn.svm import SVC
print('\nSVC')
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))