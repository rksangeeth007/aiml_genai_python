import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100.0, 20, 10000)

#plt.hist(incomes, 50)
#plt.show()

#print(np.std(incomes))
#print(np.var(incomes))


values = np.random.uniform(-10.0, 10.0, 100000)
#plt.hist(values, 50)
#plt.show()

from scipy.stats import norm

# x = np.arange(-3, 3, 0.001)
# plt.plot(x, norm.pdf(x))
# plt.show()


# mu = 5.0
# sigma = 2.0
# values = np.random.normal(mu, sigma, 10000)
# plt.hist(values, 50)
# plt.show()


# from scipy.stats import expon
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 10, 0.001)
# plt.plot(x, expon.pdf(x))
# plt.show()

# from scipy.stats import binom
# import matplotlib.pyplot as plt
#
# n, p = 10, 0.5
# x = np.arange(0, 10, 0.001)
# plt.plot(x, binom.pmf(x, n, p))
# plt.show()


# from scipy.stats import poisson
# import matplotlib.pyplot as plt
#
# mu = 500
# x = np.arange(400, 600, 0.5)
# plt.plot(x, poisson.pmf(x, mu))
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

vals = np.random.normal(10, 0.5, 10000)

plt.hist(vals, 50)
plt.show()

#print(np.percentile(vals,50))
#print(np.var(vals))
import scipy.stats as sp
print(sp.skew(vals))
print(sp.kurtosis(vals))
