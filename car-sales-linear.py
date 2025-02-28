import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("car_price_dataset.csv")

print(df.head())
print(df.info)
print(df.isnull().sum())

lb = LabelEncoder()
df['Fuel_Type'] = lb.fit_transform(df['Fuel_Type'])
df['Transmission'] = lb.fit_transform(df['Transmission'])
df['Brand'] = lb.fit_transform(df['Brand'])

df['Engine_Size'] = pd.to_numeric(df['Engine_Size'], errors='coerce')
df['Engine_Size'] = df['Engine_Size'].fillna(df['Engine_Size'].mean())
df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())
df['Price'] = df['Price'].fillna(df['Price'].mean())


x = df[['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count', 'Fuel_Type', 'Transmission', 'Brand']]
y = df['Price']  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('Coefficients:', model.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_predict))
print('R² score: %.2f' % r2_score(y_test, y_predict))

plt.scatter(y_test, y_predict, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.show()