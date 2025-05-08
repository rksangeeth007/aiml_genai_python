from sklearn.datasets import load_iris

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

import xgboost as xgb

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
            'max_depth': 4,
            'eta': 0.3,
            'objective': 'multi:softmax',
'num_class': 3}
epochs = 10

model = xgb.train(param, train, epochs)
predictions = model.predict(test)

print(predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))