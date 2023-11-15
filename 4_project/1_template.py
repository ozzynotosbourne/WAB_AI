from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=100, factor=0.2, noise=0.3)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()