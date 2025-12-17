import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('house_rent.csv')


df['age'] = df['age'].fillna(df['age'].mean())
df = df.drop_duplicates()
#print(df.isnull())

x = df[["size_sqft","bedrooms","age"]]
y = df["rent"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mdl = LinearRegression()
mdl.fit(x_train, y_train)

#prediction
y_pred = mdl.predict(x_test)

print(y_pred)

#mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(mae)