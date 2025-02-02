import numpy as np
def predict(Theta1, Theta2, X):
    m= X.shape[0]
    one_matrix= np.ones((m,1))
    X= np.append(one_matrix, X, axis= 1)
    Z2= np.dot(X, Theta1.transpose())
    A2= 1/(1+np.exp(-Z2))
    one_matrix= np.ones((m,1))
    A2= np.append(one_matrix, A2, axis=1)
    Z3= np.dot(A2, Theta2.transpose())
    A3= 1/(1+np.exp(-Z3))
    P= (np.argmax(A3, axis=1))
    return P
