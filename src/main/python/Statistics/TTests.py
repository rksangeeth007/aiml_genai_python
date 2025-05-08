import numpy as np
from scipy import stats

A = np.random.normal(25.0, 5.0, 100000)
B = np.random.normal(25.0, 5.0, 100000)

print(stats.ttest_ind(A, A))