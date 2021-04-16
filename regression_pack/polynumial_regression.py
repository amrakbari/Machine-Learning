import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, Y)

plt.scatter(X, Y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('linear regression')
plt.show()

plt.scatter(X, Y, color='red')
plt.plot(X, poly_regressor.predict(X_poly), color='blue')
plt.title('polynomial regression')
plt.show()

print('linear regression prediction of value 7.5')
print(linear_regressor.predict([[7.5]]))
print('polynomial regression prediction of value 7.5')
print(poly_regressor.predict(poly_reg.fit_transform([[7.5]])))
