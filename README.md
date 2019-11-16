# Skipgram
A feed-forward neural network to convert English words into N-dimensional vectors. Recognizes semantic closeness between different words. Developed from scratch in Python without using any machine learning library.

1. "n_net_minibatch.py" is the training file. It trains the neural network for any given dataset ("dataset.csv") and generates "skipgram_w1.npy", "initialPlot.png" (word embeddings of untrained word vectors) and "finalPlot.png" (word embeddings for trained word vectors).

2. The resultant trained word vectors  are preserved as "skipgram_w1.npy".

3. "predict.py" uses the trained word vectors to: (i) output cosine similarity between two input words. (ii) output 10 closest context words to any input words. This code has been formatted to fetch input from command line.
