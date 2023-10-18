import pandas as pd    #'as'  alias
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
print(df.head())

print(df.Gender.value_counts())

df.Height *= 2.54
df.Weight /= 2.2

print(df.head())

# sns.displot(df.query("Gender=='Male'").Weight)
# sns.displot(df.query("Gender=='Female'").Weight)
# plt.show()

df = pd.get_dummies(df)
print(df.head())
del (df["Gender_Male"])
df.rename(columns={'Gender_Female': 'Gender'}, inplace=True)
print(df.head())

model = LinearRegression()
model.fit(df[['Height', 'Gender']], df['Weight'])

print(model.coef_, model.intercept_)
print('equation: Height * ',model.coef_[0],'   + Gender * ',model.coef_[1],' + ',model.intercept_,' = Weight')

print(df.query("Gender=='Male'").Weight)

sns.scatterplot(data=df, x='Height', y='Weight')
plt.show()