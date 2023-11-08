import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# data preparation
df = pd.read_csv('weight-height.csv')
# print(df)
print(df.head(10))

print(df.Gender.value_counts())   #check now many men and women we have

# df.Height = df.Height * 2.54
df.Height *= 2.54
df.Weight /= 2.2
print(df.head(10))

# sns.displot(df.Weight)   #men and women together
sns.displot(df.query("Gender=='Male'").Weight)
sns.displot(df.query("Gender=='Female'").Weight)
plt.show()

df = pd.get_dummies(df)   #change gender to 2 columns
del(df["Gender_Male"])    #delete one column
df.rename(columns={"Gender_Female":"Gender"}, inplace=True)   #change name
print(df)
# data is ready


# algorithm
model = LinearRegression()
model.fit(df[["Height","Gender"]],df["Weight"])
print(model.coef_, model.intercept_)
