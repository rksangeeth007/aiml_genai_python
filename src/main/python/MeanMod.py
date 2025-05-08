import numpy as np

incomes = np.random.normal(27000, 15000, 10000)
#print(np.mean(incomes))

import matplotlib.pyplot as plt
plt.hist(incomes, 50)
#plt.show()

#print(np.median(incomes))

ages = np.random.randint(18,90,500)
from scipy import stats
print(stats.mode(ages))