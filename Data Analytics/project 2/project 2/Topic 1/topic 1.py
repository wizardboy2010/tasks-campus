import pandas as pd
import numpy as np
data = pd.ExcelFile('MOVIE.xlsx').parse()

pop_temp = data.iloc[1:, [-5,-3]].values.astype(float)
pop = []

for i in range(len(pop_temp[:,0])):
    if pop_temp[i,0]<=2015:
        pop.append(pop_temp[i,:])

pop_mean = np.mean(np.asarray(pop)[:,1])
print("Population mean of imdb score is", pop_mean)

sample = []
for i in range(len(pop_temp[:,0])):
    if pop_temp[i,0] == 2016:
        sample.append(pop_temp[i,1])
sample = np.array(sample)
