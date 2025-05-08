def sqr(x):
    return x*x

def doSomething(f,x):
    return f(x)
#
# print(doSomething(sqr,4))
# print(doSomething(lambda x: x*x*x , 5))
#
# for i in range(10):
#     print(i)

# i = 0
# while(i <= 10):
#     print(i)
#     i+=1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.plot(range(10))
# plt.savefig("myplot.png")
# plt.show()

df = pd.read_csv("/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/files/PastHires.csv")
#print(df.sort_values(['Years Experience']))
#print(df['Employed?'].value_counts())

# degree_count = df['Level of Education'].value_counts()
# degree_count.plot(kind = 'bar')
# plt.show()

exer_df = df[['Previous employers', 'Hired']]
exer_df.plot(kind = 'hist')
plt.show()