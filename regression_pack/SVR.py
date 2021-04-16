import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values
Y = Y.reshape(len(Y), 1)
FeaturesSC = StandardScaler()
DependentSC = StandardScaler()
X = FeaturesSC.fit_transform(X)
Y = DependentSC.fit_transform(Y)
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)
Y_pred = regressor.predict(FeaturesSC.transform([[6.5]]))
Y_pred = DependentSC.inverse_transform(Y_pred)
print('SVR regression prediction of 6.5:')
print(Y_pred)

plt.scatter(FeaturesSC.inverse_transform(X), DependentSC.inverse_transform(Y), color='red')
plt.plot(FeaturesSC.inverse_transform(X), DependentSC.inverse_transform(regressor.predict(X)), color='blue')
plt.title('SVR regression')
plt.show()
