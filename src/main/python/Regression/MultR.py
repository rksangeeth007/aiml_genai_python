import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

import numpy as np
df1=df[['Mileage','Price']]
bins =  np.arange(0,50000,10000)
groups = df1.groupby(pd.cut(df1['Mileage'],bins)).mean()
print(groups.head())
groups['Price'].plot.line()
#plt.show()

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values)

X = sm.add_constant(X)
#print (X)

est = sm.OLS(y, X).fit()
#print(est.summary())

#print(y.groupby(df.Doors).mean())

scaled = scale.transform([[45000, 8, 4]])
scaled = np.insert(scaled[0], 0, 1)
print(scaled)
predicted = est.predict(scaled)
print(predicted)
