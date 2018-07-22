import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.ExcelFile('SALARY.xlsx').parse()

x = data[(data['Year']==2014)].iloc[:,3:7]
x = x[x.BasePay != 'Not Provided']

x1 = np.array(x).astype(float)

corr = np.corrcoef(x1.T)

print("correlation matrix: ", corr)

plt.scatter(x1[:,0], x1[:,-1])
plt.xlabel('BasePay')
plt.ylabel('Benfifits')

plt.scatter(x1[:,0], x1[:,-2])
plt.xlabel('BasePay')
plt.ylabel('OtherPay')

plt.scatter(x1[:,0], x1[:,-3])
plt.xlabel('BasePay')
plt.ylabel('OvertimePay')
