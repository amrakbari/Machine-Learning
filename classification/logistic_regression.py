import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('../data/Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
# y_pred = classifier.predict(sc.transform([[30, 87000]]))
y_pred = classifier.predict((x_test))
# y_pred_reshaped = y_pred.reshape(len(y_pred), 1)
# y_test_reshaped = y_test.reshape(len(y_test), 1)
# print(np.concatenate((y_pred_reshaped, y_test_reshaped), 1))

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)