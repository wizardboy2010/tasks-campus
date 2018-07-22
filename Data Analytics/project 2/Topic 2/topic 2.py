import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.ExcelFile('NUTRITION.xlsx').parse()

correlation = np.corrcoef(data.iloc[1:,9].values.astype(int), data.iloc[1:,-1].values.astype(float))[0, 1]
print('correlation coefficient is:', correlation)

#plt.scatter(data.iloc[1:,9].values.astype(int),data.iloc[1:,-1].values.astype(float))

x = data.iloc[1:,9].values.astype(float).reshape(77,1)
y = data.iloc[1:,-1].values.astype(float).reshape(77,1)
from sklearn.preprocessing import PolynomialFeatures
deg2 = PolynomialFeatures(degree = 2)
deg3 = PolynomialFeatures(degree = 3)

x2 = deg2.fit_transform(x)
x3 = deg3.fit_transform(x)

from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()
reg2 = LinearRegression()
reg3 = LinearRegression()
reg1.fit(x,y)
reg2.fit(x2,y)
reg3.fit(x3,y)

score = []
from sklearn.metrics import r2_score
score.append(r2_score(y, reg1.predict(x)))
score.append(r2_score(y, reg2.predict(x2)))
score.append(r2_score(y, reg3.predict(x3)))

print("Maximum R score is:", max(score),"for a degree of ", np.argmax(score)+1)

print("We can say that they are non linearly correlated with order 3 with R score of 0.609234412881")