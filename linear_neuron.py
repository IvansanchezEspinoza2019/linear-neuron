import numpy as np;
"""
    This neuron it's a modification of the classic perceptron. The  per-
    ceptron only do binaries classifications, meanwhile this neuron do 
    linear regression. This is because the preceptron has an activation
    function that makes discret outputs, the linear neuron delets this f-
    unction to make a continuos output.'
"""

import matplotlib.pyplot as plt

class LinearNeuron:
    def __init__(self, n_dim, learn_fact):
        self.w = -1 + 2 * np.random.rand(n_dim)  # weights vector
        self.b = -1 + 2 * np.random.rand()       # bias variable
        self.eta = learn_fact                    # neuron learning factor
        
    def predict(self, X):                        # make predictions
        Y_predict = np.dot(self.w, X) + self.b
        return Y_predict
        
    def fit(self, X, Y, solver='SGD', epochs=100):      # train the neuron
        p = X.shape[1]              # patterns
        
        if solver == 'SGD':         # STOCASTIC GRADIENT DESCEND
            for _ in range(epochs):
                for i in range(p):
                    y_pred = self.predict(X[:, i])
                    self.w += self.eta * (Y[:, i] - y_pred) * X[:, i]
                    self.b += self.eta * (Y[:, i] - y_pred)
                    
        elif solver == 'BGD':     # BATCH GRADIENT DESCEND
            for _ in range(epochs):
                Y_pred = np.dot(self.w, X) + self.b
                self.w += (self.eta/p) * np.dot((Y - Y_pred), X.T).ravel()
                self.b += (self.eta/p) * np.sum(Y - Y_pred)
                
        else:                     # DIRECT METHOD
            ones = np.ones((1,p))
            x_hom = np.concatenate((ones, X), axis=0 )                  # concatenate a vector of ones
            w_hom = np.dot(Y.reshape(1, -1), np.linalg.pinv(x_hom))     # do the pseudo-inverse of X multiplied by Y
            self.b = w_hom[0, 0]                                        # get the bias value
            self.w = w_hom[0, 1:]                                       # get the final weights
                
            

            
            
        
