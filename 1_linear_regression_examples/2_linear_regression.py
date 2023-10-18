import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('otodom.csv')
print(df.head(10).to_string())
print(df.describe().T.to_string())

print(df.iloc[:,1:].corr())   #korelacja
sns.heatmap( df.iloc[:,1:].corr(), annot=True )
plt.show()
sns.displot(df.cena)
plt.show()
plt.scatter(df.powierzchnia, df.cena)
plt.show()
print(df.describe())

_min = df.describe().loc['min','cena']
q1 = df.describe().loc['25%','cena']
q3 = df.describe().loc['75%','cena']
print(_min, q1, q3)

df1 = df[ (df.cena >= _min) & (df.cena <= q3) & (df.rok_budowy < df.describe().loc['max','rok_budowy'])]
#sns.displot(df1.cena)
#plt.show()
print('nowy describe')
print(df1.describe().to_string())
#podział na dane treningowe i testowe
print('teraz dane')
print(df1.columns)
X = df1.iloc[:, 2: ]   #liczba_pieter  liczba_pokoi  pietro  powierzchnia  rok_budowy
y = df1.cena
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#wybór i trenowanie modelu
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.coef_)
