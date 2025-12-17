import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

df = pd.read_csv('battery_life.csv')
df.drop_duplicates()
#print(df)

#print(df.isnull())

#step transform
label_encoder = preprocessing.LabelEncoder()
label = label_encoder.fit(df['network'])
df['network']= label_encoder.transform(df['network'])

#print(df)

x = df[['screen_time','apps_used','network']]
y = df['battery_hours']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mdl = LinearRegression()
mdl.fit(x_train, y_train)

#prediction
y_pred = mdl.predict(x_test)

result = x_test.copy()
result['battery_hours'] = y_test
result['predicted_batteryhours'] = y_pred
print(result)

print(mean_absolute_error(y_test, y_pred))
