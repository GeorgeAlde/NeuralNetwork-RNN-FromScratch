import numpy as np

from RNNdataset import train_data


def createinputs(text):
    inputs = []
    for word in word_to_idx:
        v = np.zeros((vocab_size, 1))
        print(v)
        v[word_to_idx[word]] = 1
        inputs.append(v)
    return inputs


vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
# print(f'{vocab_size} unique words found')

# assign indices to each word
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# print(word_to_idx)
# print(idx_to_word)



