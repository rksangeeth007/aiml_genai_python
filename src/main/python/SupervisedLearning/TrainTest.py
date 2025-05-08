import matplotlib.pyplot as plt
import numpy as np
from pylab import *

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds

#scatter(pageSpeeds, purchaseAmount)
#plt.show()


trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

#scatter(trainX, trainY)
#scatter(testX, testY)
#plt.show()

x = np.array(trainX)
y = np.array(trainY)
p4 = np.poly1d(np.polyfit(x, y, 8))

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
# plt.scatter(x, y)
# plt.plot(xp, p4(xp), c='r')
# plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(trainY, p4(trainX))
print(r2)

testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.show()

from sklearn.metrics import r2_score
r2 = r2_score(testy, p4(testx))
print(r2)
