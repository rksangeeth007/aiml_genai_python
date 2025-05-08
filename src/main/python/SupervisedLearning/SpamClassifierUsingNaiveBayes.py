import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = pd.concat([data, dataFrameFromDirectory("/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/emails/spam", "spam")]);
data = pd.concat([data, dataFrameFromDirectory("/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/emails/ham", "ham")])

#print(data.head())

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
MultinomialNB()

examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
print("example_counts" + str(example_counts))
predictions = classifier.predict(example_counts)
print("predictions" + str(predictions))
