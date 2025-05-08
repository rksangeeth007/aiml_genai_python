#This can be executed in IJ due to dependency issues & GPU req, Instead use Google Collab

from transformers import BertModel, BertTokenizer

modelName = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(modelName)

model = BertModel.from_pretrained(modelName)

tokenized = tokenizer("I read a good novel.")
print(tokenized)

tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
print(tokens)

#CLS is a token which says the starting point. SEP is the end.

# Positional Encoding

import numpy as np
import matplotlib.pyplot as plt

def encodePositions(num_tokens, depth, n=10000):
    positionalMatrix = np.zeros((num_tokens, depth))
    for row in range(num_tokens):
        for col in np.arange(int(depth/2)):
            denominator = np.power(n, 2*col/depth)
            positionalMatrix[row, 2*col] = np.sin(row/denominator)
            positionalMatrix[row, 2*col+1] = np.cos(row/denominator)
    return positionalMatrix

positionalMatrix = encodePositions(50, 256)
fig = plt.matshow(positionalMatrix)
plt.gcf().colorbar(fig)

# Self-Attention


# from bertviz.transformers_neuron_view import BertModel, BertTokenizer
# from bertviz.neuron_view import show
#
# tokenizer_viz = BertTokenizer.from_pretrained(modelName)
# model_viz = BertModel.from_pretrained(modelName)
# show(model_viz, "bert", tokenizer_viz, "I read a good novel.", display_mode="light", head=11)
#
# show(model_viz, "bert", tokenizer_viz, "Attention is a novel idea.", display_mode="light", head=11)
