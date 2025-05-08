import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
df = pd.read_csv('BostonHousingData.csv')
df.head()
df.info()
df.duplicated()
df.isnull().sum()
for col in ['CRIM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']:
    df[col] = df[col].fillna(df[col].mean())
df.isnull().sum()
y = df['MEDV']
X = df.drop(columns=['MEDV'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = sqrt(mse_train)
rmse_test = sqrt(mse_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
print(f"Traning: \n MSE : {mse_train:.2f}, R2 : {r2_train:.2f}, RMSE : {rmse_train:.2f}, MAE : {mae_train:.2f}")
print(f"Testing: \n MSE : {mse_test:.2f}, R2 : {r2_test:.2f}, RMSE : {rmse_test:.2f}, MAE : {mae_test:.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted House Prices')
plt.show()