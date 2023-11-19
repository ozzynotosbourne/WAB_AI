from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Generate data using make_circles
X, y = make_circles(n_samples=1000, factor=0.2, noise=0.4)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic Regression
print('\nLogistic Regression')
model_lr = LogisticRegression(solver='liblinear', max_iter=5000, multi_class='ovr', penalty='l2')
model_lr.fit(X_train, y_train)
print('Accuracy:', model_lr.score(X_test, y_test))
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y_test, model_lr.predict(X_test))))

# K-Nearest Neighbors (KNN)
print('\nKNN')
model_knn = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1, p=1)
model_knn.fit(X_train, y_train)
print('Accuracy:', model_knn.score(X_test, y_test))
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y_test, model_knn.predict(X_test))))

# Decision Tree Classifier
print('\nDecision Tree Classifier')
model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=10, min_samples_leaf=5)
model_dt.fit(X_train, y_train)
print('Accuracy:', model_dt.score(X_test, y_test))
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y_test, model_dt.predict(X_test))))

# Support Vector Machine (SVC)
print('\nSVC')
model_svc = SVC(kernel='rbf', degree=3, gamma='scale', decision_function_shape='ovo', C=0.5)
model_svc.fit(X_train, y_train)
print('Accuracy:', model_svc.score(X_test, y_test))
print('Confusion Matrix:')
print(pd.DataFrame(confusion_matrix(y_test, model_svc.predict(X_test))))
