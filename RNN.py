import numpy as np
from numpy.random import randn #randn is from the normal distribution

from RNNdataset import train_data


def createinputs(text):
    inputs = []
    for word in text.split(' '):
        v = np.zeros((vocab_size, 1))
        #print(v)
        v[word_to_idx[word]] = 1
        inputs.append(v)
    return inputs


def softmax(xs): #xs is an array
    return np.exp(xs) / sum(np.exp(xs))


vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
# print(f'{vocab_size} unique words found')

# assign indices to each word
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# print(word_to_idx)
# print(idx_to_word)

class RNN:

    def __init__(self, input_size, output_size, hidden_size = 64):
        #Weights (dividing by 1000 to reduce variance)
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        #Biases
        #np.zeros creates array of (rows, columns) with zeros
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        #shape gives dimensions of a matrix ie (64, 64)
        h = np.zeros((self.Whh.shape[0], 1))

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

        y = self.Why @ h + self.by

        return y, h

rnn = RNN(vocab_size, 2)

inputs = createinputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)




