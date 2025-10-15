import numpy as np
from numpy.random import randn  # randn is from the normal distribution
from random import shuffle

from RNNdataset import train_data, test_data


def createinputs(text):
    inputs = []
    for word in text.split(' '):
        v = np.zeros((vocab_size, 1))
        # print(v)
        v[word_to_idx[word]] = 1
        inputs.append(v)
    return inputs


def softmax(xs):  # xs is an array
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

    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights (dividing by 1000 to reduce variance)
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        # np.zeros creates array of (rows, columns) with zeros
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        # shape gives dimensions of a matrix ie (64, 64)
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=0.02):
        n = len(self.last_inputs)

        dL_Why = d_y @ self.last_hs[n].T
        dL_by = d_y

        dL_Whh = np.zeros(self.Whh.shape)
        dL_Wxh = np.zeros(self.Wxh.shape)
        dL_bh = np.zeros(self.bh.shape)

        dL_h = self.Why.T @ d_y

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t+1] ** 2) * dL_h)

            dL_bh += temp
            dL_Whh += temp @ self.last_hs[t].T
            dL_Wxh += temp @ self.last_inputs[t].T
            dL_h = self.Whh @ temp

            for d in [dL_Wxh, dL_Whh, dL_Why, dL_by, dL_by]:
                np.clip(d, -1, 1, out=d)

        self.Whh -= learn_rate * dL_Whh
        self.Wxh -= learn_rate * dL_Wxh
        self.Why -= learn_rate * dL_Why
        self.bh -= learn_rate * dL_bh
        self.by -= learn_rate * dL_by



rnn = RNN(vocab_size, 2)
def processData(data, backprop=True):
    items = list(data.items())
    shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createinputs(x)
        target = int(y)

        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            dL_dy = probs
            dL_dy[target] -= 1

            rnn.backprop(dL_dy)

    return loss / len(data), num_correct / len(data)

for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch +1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

inputs = createinputs('i am not very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)