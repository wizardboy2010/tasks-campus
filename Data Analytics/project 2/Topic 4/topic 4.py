import pandas as pd
import numpy as np

data = pd.ExcelFile('SNACKS.xls').parse()

from scipy.stats import spearmanr

s, p = spearmanr(data.iloc[:,:])

from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()
x = oe.fit_transform(data.iloc[:,:]).toarray()

y = x[:,0:9]
x = x[:,9:]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial')
model.fit(x,y)