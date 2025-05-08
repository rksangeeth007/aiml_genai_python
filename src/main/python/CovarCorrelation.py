import numpy as np
from pylab import *

def de_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
#purchaseAmount = np.random.normal(50.0, 10.0, 1000)

# scatter(pageSpeeds, purchaseAmount)
# plt.show()
#print(covariance (pageSpeeds, purchaseAmount))


#purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
#scatter(pageSpeeds, purchaseAmount)
#plt.show()

#print(covariance (pageSpeeds, purchaseAmount))


def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y) / stddevx / stddevy  #In real life you'd check for divide by zero here

#print(correlation(pageSpeeds, purchaseAmount))

#print(np.corrcoef(pageSpeeds, purchaseAmount))

purchaseAmount = 100 - pageSpeeds * 3
#scatter(pageSpeeds, purchaseAmount)
#plt.show()
#print(correlation (pageSpeeds, purchaseAmount))


#print(np.cov(pageSpeeds, purchaseAmount))
print(np.correlate(pageSpeeds, purchaseAmount))
