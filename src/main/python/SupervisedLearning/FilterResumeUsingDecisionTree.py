import numpy as np
import pandas as pd
from sklearn import tree

input_file = "/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/PastHires.csv"
df = pd.read_csv(input_file, header = 0)
print(df.head())

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())

features = list(df.columns[:6])

y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

from IPython.display import Image
from io import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())    #Not coming up on the IJ
