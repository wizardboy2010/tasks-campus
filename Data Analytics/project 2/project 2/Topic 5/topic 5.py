# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

data = pd.ExcelFile('GAMES.xlsx').parse()

x = data.iloc[:,[3,-4,-1]]
x = x.dropna(axis = 0)
x = x[x.User_Score != 'tbd']

a = np.mean((x[x.Rating == 'T'])[x[x.Rating == 'T'].Genre == 'Action'].iloc[:,1].values.astype(float))
b = np.mean((x[x.Rating != 'T'])[x[x.Rating != 'T'].Genre == 'Action'].iloc[:,1].values.astype(float))
c = np.mean((x[x.Rating == 'T'])[x[x.Rating == 'T'].Genre != 'Action'].iloc[:,1].values.astype(float))
d = np.mean((x[x.Rating != 'T'])[x[x.Rating != 'T'].Genre != 'Action'].iloc[:,1].values.astype(float))

obs = np.array([[a,b],[c,d]])
print("\n   observed contingency table:\n", obs)

a1 = (a+b)*(a+c)/(a+b+c+d)
b1 = (a+b)*(b+d)/(a+b+c+d)
c1 = (c+d)*(a+c)/(a+b+c+d)
d1 = (c+d)*(b+d)/(a+b+c+d)

exp = np.array([[a1,b1],[c1,d1]])
print("\n expected contingency table:\n", exp)

xchi2 = np.sum((obs-exp)**2/exp)
print("\n X2 value is",xchi2)
