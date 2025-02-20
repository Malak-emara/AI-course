
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("Salary_Data.csv")  

threshold = df['Salary'].median()
df['Salary_Class'] = (df['Salary'] >= threshold).astype(int)  # 0 = Low Salary, 1 = High Salary

X = df[['YearsExperience']]
y = df['Salary_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', marker='x')
plt.xlabel("Years of Experience")
plt.ylabel("Salary Class (0 = Low, 1 = High)")
plt.title("Logistic Regression: Salary Classification")
plt.legend()
plt.show()
