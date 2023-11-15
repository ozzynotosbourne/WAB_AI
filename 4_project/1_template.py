from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

X, y = make_circles(n_samples=50, factor=0.2, noise=0.4)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
print('\nLogistic regression')
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))