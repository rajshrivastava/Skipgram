# Skipgram
Skip-gram recognizes semantic closeness between different words by converting English words into N-dimensional vectors. The idea was proposed by Mikolov et al. in https://arxiv.org/abs/1301.3781. The original code was written in C.

This repository contains the implementation of neural network for skip-gram from scratch in Python without using any machine learning or text processing libraries.

1. "n_net_minibatch.py" is the training file. It trains the neural network for any given dataset ("dataset.csv") and generates "skipgram_w1.npy", "initialPlot.png" (word embeddings of untrained word vectors) and "finalPlot.png" (word embeddings for trained word vectors).

2. The resultant trained word vectors  are preserved as "skipgram_w1.npy".

3. "predict.py" uses the trained word vectors to: (i) output cosine similarity between two input words. (ii) output 10 closest context words to any input words. This code has been formatted to fetch input from command line.

Also, here is another implementation [Using TensorFlow](https://github.com/rajshrivastava/Word-embeddings).
