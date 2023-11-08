import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")
print(df)

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
print(df)
X = df.iloc[:, :4]  # take 4 first columns
y = df.class_value
model = DecisionTreeClassifier()
model.fit(X, y)

from dtreeplt import dtreeplt
dtreeplt(model)
plt.show()