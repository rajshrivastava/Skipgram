import numpy as np
import pickle
import time
import pandas as pd
from sklearn.decomposition import PCA #sklearn's PCA has been used just for visualizing the embeddings in two dimensions
from matplotlib import pyplot

class NeuralNet():
    def __init__(self, config):                           #initializing hyperparameters
        self.n = config['n']
        self.eta = config['eta']
        self.epochs = config['epochs']
        self.window = config['window_size']     #A random window size within a range would yield better results.
        self.mini_batch_size=config['mini_batch_size']
        pass
    
    def create_corpus(self, tweets):                      #data cleansing
        sentences=list(tweets)
        stop_words=['k','m','t','d','e','f','g','h','i','u','r','I','im','ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'should', 'our', 'their', 'while', 'above', 'both', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'];
        filtered_sentences=[]     #sentences without stop_words
        for i in range(len(sentences)):
            sentence=sentences[i]
            if(type(sentence)== float):
                continue
            temp=sentence.split()
    
            filtered_words=[]
            for w in temp:
                if w not in stop_words:
                    filtered_words.append(w)
            filtered_sentences.append(filtered_words)
                
        corpus=[]    #filtered sentences minus sentences<2
        for i, sentence in enumerate(filtered_sentences):
            if(len(sentence)<2):
                continue
            else:
                corpus.append(sentence)
                
        return corpus
    
    def word2hot(self, word):                             #onehot encoding for word
        onehot = [0 for i in range(0, self.voc_size)]
        word_index = self.word_index[word]
        onehot[word_index] = 1
        return onehot
    
    def format_data(self,corpus):                         #preparing data for training                             
        total_words=0
        word_counts = {}   #dictionary to store the words and their counts
        for sentence in corpus:
            for word in sentence:
                if word in word_counts:
                    word_counts[word] += 1   #for existing words
                else:
                    word_counts[word]= 1     #for new words
                total_words += 1
        self.voc_size = len(word_counts.keys())
        print("Length of vocabulary = ",self.voc_size)
        print("Total words = " , total_words)
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        
        self.w1   = np.random.randn(self.n, self.voc_size)    #input -> hidden
        self.w2   = np.random.randn(self.voc_size, self.n)    #hidden -> output
        
        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):                
                w_target = self.word2hot(word)
                w_context = []
                for j in range(i-self.window, i+self.window+1):         
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word_index[sentence[j]])
                training_data.append([w_target, w_context])
        return np.array(training_data)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    
    def forward_pass(self, x):
        h = np.dot(self.w1, x)
        u = np.dot(self.w2, h)
        y_pred = self.softmax(u)
        return y_pred, h, u
                
    def logloss_prime(self, delta, w_c):
        for word in w_c:
            delta[word] = delta[word] - 1   #derivative of negative log likelihood with respect to the softmax layer for each context
        return delta                        
    
    def backprop(self, w_t, w_c):
        y_pred, h, u = self.forward_pass(w_t)       
        
        delta_out =  self.logloss_prime(y_pred, w_c)   
        
        nabla_w2  = np.outer(delta_out, h) 
        delta_h   = np.dot(self.w2.T, delta_out)
        nabla_w1 = np.outer(delta_h, w_t)
        
        total_loss= np.sum(abs(delta_out))      #loss with respect to all the contexts
        
        return total_loss, nabla_w1, nabla_w2
    
    def update_mini_batch(self, mini_batch):
        nabla_w1   = np.zeros(self.w1.shape)
        nabla_w2   = np.zeros(self.w2.shape)
        nabla_loss=0
        self.sample_index=0
        for w_t, w_c in mini_batch:
            #print("batch_index",self.batch_index,"..sample_index",self.sample_index)
            loss, delta_nabla_w1, delta_nabla_w2 = self.backprop(w_t, w_c)
            #summing of nabla weights and biases of each sample in the mini-batch
            nabla_w1   = nabla_w1   + delta_nabla_w1
            nabla_w2   = nabla_w2   + delta_nabla_w2
            nabla_loss += loss
            self.sample_index += 1
            
        #weights and bias updations (does not return anything)
        self.w1   = self.w1   - (self.eta/self.sample_index)*nabla_w1
        self.w2   = self.w2   - (self.eta/self.sample_index)*nabla_w2
        
        return nabla_loss/self.sample_index
    
    def train(self, training_data):
        n = len(training_data)
        print("Length of training data: ", n,". Learing rate: ", self.eta)        
        print("Training started at ", time.ctime(time.time()) )
        for epo in range(self.epochs):
            np.random.shuffle(training_data)
            mini_batches = [ training_data[k:k+self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
            epoch_loss = 0
            self.batch_index = 0
            for mini_batch in mini_batches:
                batch_loss = self.update_mini_batch(mini_batch)
                self.batch_index += 1         
                epoch_loss += batch_loss
            print("Epoch ", epo+1, " of ", self.epochs, " completed at ",time.ctime(time.time()), ". Loss = ", epoch_loss/self.batch_index)
        print("Training ended at ", time.ctime(time.time()) )
        
    def save_model(self):
        np.save('skipgram_w1.npy', self.w1)
        
        out_file=open('skipgram_word_index.txt','wb')
        pickle.dump(self.word_index,out_file)
        out_file.close()
        
        out_file=open('skipgram_index_word.txt','wb')
        pickle.dump(self.index_word,out_file)
        out_file.close()        
    
    def visual(self, name):
        pca = PCA(n_components=2)
        result = pca.fit_transform(self.w1.T)
        pyplot.scatter(result[:, 0], result[:, 1])
        words = list(self.word_index.keys())
        for i, word in enumerate(words):
       	    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.savefig(name)
        pyplot.show()
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
    config={'n':50, 'window_size':2, 'mini_batch_size':32, 'epochs':20, 'eta':0.1}
    
    data=pd.read_csv('dataset.csv')
    tweets   = data.iloc[795000:805000,1]

    model = NeuralNet(config)    
    corpus= model.create_corpus(tweets)
    print("Total no. of tweets: ", len(corpus))
    training_data = model.format_data(corpus)        #each element of training data is of the form: [ target_onehot, [context1_index, context2_index,.....] ]    
    model.visual("initialPlot.png")
    model.train(training_data)      #training neural network
    model.visual("finalPlot.png")
    model.save_model()              #saving word vectors to file