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
model = LogisticRegression(solver='liblinear', max_iter=2000, multi_class='ovr', verbose=0)   #choose the model
model.fit(X_train, y_train)    #teach the model
print(model.score(X_test, y_test))      #print accuracy
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))    #print details

from sklearn.neighbors import KNeighborsClassifier
print('\nKNN')
model = KNeighborsClassifier(n_neighbors=530, weights='distance', n_jobs=10)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from sklearn.tree import DecisionTreeClassifier
print('\nDecission  Tree Clasifier')
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from sklearn.svm import SVC
print('\nSVC')
model = SVC(kernel='poly', degree=10, gamma='auto', decision_function_shape='ovo')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

# ................. project
# choose "make_circles" parameters and generate your data
# find possibly best sulution using at least 2 algoritms algorthms
# investigate parameters mentioned in example
# add at least one more parameter to each algorithm (not verbose, not random state)
# deliver code + pdf report on moodle till day of our last class