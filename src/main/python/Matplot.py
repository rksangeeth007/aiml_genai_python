import random

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3, 3, 0.01)

# plt.plot(x, norm.pdf(x))
# plt.plot(x, norm.pdf(x, 1.0, 0.5))
#plt.show()
#plt.savefig('Myplot.png', format='png')

# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# plt.plot(x, norm.pdf(x))
# plt.plot(x, norm.pdf(x, 1.0, 0.5))
# plt.show()


# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# axes.grid()
# plt.plot(x, norm.pdf(x))
# plt.plot(x, norm.pdf(x, 1.0, 0.5))
# plt.show()

# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# axes.grid()
# plt.plot(x, norm.pdf(x), 'b-')
# plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r--')
# plt.show()

# axes = plt.axes()
# axes.set_xlim([-5, 5])
# axes.set_ylim([0, 1.0])
# axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# axes.grid()
# plt.xlabel('Greebles')
# plt.ylabel('Probability')
# plt.plot(x, norm.pdf(x), 'b-')
# plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')
# plt.legend(['Sneetches', 'Gacks'], loc=4)
# plt.show()

# plt.xkcd()
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# plt.xticks([])
# plt.yticks([])
# ax.set_ylim([-30, 10])
#
# data = np.ones(100)
# data[70:] -= np.arange(30)
#
# plt.annotate(
#     'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
#     xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))
#
# plt.plot(data)
#
# plt.xlabel('time')
# plt.ylabel('my overall health')
# plt.show()


#plt.rcdefaults()

# values = [12, 55, 4, 32, 14]
# colors = ['r', 'g', 'b', 'c', 'm']
# explode = [0, 0, 0.2, 0, 0]
# labels = ['India', 'United States', 'Russia', 'China', 'Europe']
# #plt.pie(values, colors= colors, labels=labels, explode = explode)
# plt.bar(range(0,5),values, colcolors)
# plt.title('Student Locations')
# plt.show()

#values = [12, 55, 4, 32, 14]
# colors = ['r', 'g', 'b', 'c', 'm']
# plt.bar(range(0,5), values, color= colors)
# plt.show()

# from pylab import randn
#
# X = randn(500)
# Y = randn(500)
# plt.scatter(X,Y)
# plt.show()

# incomes = np.random.normal(27000, 15000, 10000)
# plt.hist(incomes, 50)
# plt.show()

# uniformSkewed = np.random.rand(100) * 100 - 40
# high_outliers = np.random.rand(10) * 50 + 100
# low_outliers = np.random.rand(10) * -50 - 100
# data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
# plt.boxplot(data)
# plt.show()


import pandas as pd
df = pd.read_csv("http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv")
# #print(df.head(5))
#gear_counts = df['# Gears'].value_counts()
# #print(gear_counts)
# gear_counts.plot(kind='bar')
# plt.show()

import seaborn as sns
# sns.set()
# sns.distplot(df['CombMPG'])
# #gear_counts.plot(kind='bar')
# plt.show()

# df2 = df[['Cylinders', 'CityMPG', 'HwyMPG', 'CombMPG']]
# print(df2.head())
#
# sns.pairplot(df2,height=2.5)

# sns.scatterplot(x="Eng Displ", y="CombMPG", data=df)
# sns.jointplot(x="Eng Displ", y="CombMPG", data=df)
#sns.lmplot(x="Eng Displ", y="CombMPG", data=df)
# plt.show()


# sns.set(rc={'figure.figsize':(15,5)})
# ax=sns.boxplot(x='Mfr Name', y='CombMPG', data=df)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
# plt.show()

# ax=sns.swarmplot(x='Mfr Name', y='CombMPG', data=df)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
# plt.show()

# ax=sns.countplot(x='Mfr Name', data=df)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

df2 = df.pivot_table(index='Cylinders', columns='Eng Displ', values='CombMPG', aggfunc='mean')
sns.heatmap(df2)
plt.show()
