import os
import pathlib
import numpy as np
import pandas as pd
folder='AE_weights_and_biasses'
class AE_Network:
    def __init__(self, X, hidden=[64], activation='sigmoid'):
        self.X = X
        self.W = []
        self.b = []
        self.activation=activation
        for i in range(len(hidden)+1):
            self.W.append(np.array(pd.read_csv('%s/weights_%d_%d.csv'%(folder, hidden[0],i), header=None, delimiter=','), dtype=float))
            self.b.append(np.array(pd.read_csv('%s/biasses_%d_%d.csv'%(folder, hidden[0],i), header=None, delimiter=','), dtype=float))

    def forward_propagation(self,):
        self.hidden_layers = []
        self.hidden_layers.append(activation_function(self.activation, (np.dot(self.X, self.W[0]) + self.b[0])))
        for i in range(len(self.W)//2-1):
            self.hidden_layers.append(
                activation_function(self.activation, (np.dot(self.hidden_layers[i], self.W[i + 1]) + self.b[i + 1])))
        return np.rint(self.hidden_layers[-1]).astype(int)


def activation_function(activation, x):
    if activation == 'sigmoid':
        x[(x / 700) > 1] = 700
        x[(x / 700) < -1] = -700
        return 1 / (1 + np.exp(-x))

def activation_grad(activation, x):
    x = np.array(x)
    if activation == 'sigmoid':
        return (x) * (1 - x)

def generate_hash_codes(X,hidden):
    ae=AE_Network(X, hidden)
    hash_codes=ae.forward_propagation()
    return hash_codes

