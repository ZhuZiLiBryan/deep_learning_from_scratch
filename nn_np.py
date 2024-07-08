# Some equivalent implementations using numpy library
import numpy as np

ih_weights = np.array([[0.1, 0.2, -0.1],
                       [-0.1, 0.1, 0.9], 
                       [0.1, 0.4, 0.1] ]).T

hp_weights = np.array([[0.3, 1.1, -0.3],
                       [0.1, 0.2, 0.0], 
                       [0.0, 1.3, 0.1] ]).T

weights = [ih_weights, hp_weights]


def neural_network(input, weights):
    hidden = input @ weights[0]
    pred = hidden @ weights[1]
    return pred


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

obs = 0
input = [ toes[obs], wlrec[obs], nfans[obs] ]
print(neural_network(input, weights))