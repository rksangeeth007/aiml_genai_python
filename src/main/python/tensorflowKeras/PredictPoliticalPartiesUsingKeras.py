import pandas as pd

feature_names =  ['party','handicapped-infants', 'water-project-cost-sharing',
                  'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                  'el-salvador-aid', 'religious-groups-in-schools',
                  'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                  'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                  'education-spending', 'superfund-right-to-sue', 'crime',
                  'duty-free-exports', 'export-administration-act-south-africa']

voting_data = pd.read_csv('/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/house-votes-84.data.txt', na_values=['?'],
                          names = feature_names)
#print(voting_data.head())
# print(voting_data.describe())

voting_data.dropna(inplace=True)
# print(voting_data.describe() )

voting_data.replace(('y', 'n'), (1, 0), inplace=True)
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)

print(voting_data.head())

all_features = voting_data[feature_names].drop('party', axis=1).values
all_classes = voting_data['party'].values


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

def create_model():
    model = Sequential([
        Dense(32, input_dim=16, kernel_initializer='normal', activation='relu'),
        Dense(16, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap our Keras model with SciKeras KerasClassifier
estimator = KerasClassifier(model=create_model, epochs=100, verbose=0)

# Assuming all_features and all_classes are defined and properly preprocessed
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())

