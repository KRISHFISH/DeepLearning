import numpy as np
def neural_network(nn_parms, input_layer_size, hidden_layer_size, num_labels, X, Y, lamb):
    Theta1= np.reshape(nn_parms[:hidden_layer_size * (input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2= np.reshape(nn_parms[hidden_layer_size * (input_layer_size+1):],(num_labels, hidden_layer_size+1))
    m= X.shape[0]
    one_matrix= np.ones((m,1))
    X= np.append(one_matrix, X, axis=1)
    a1= X
    z2= np.dot(X, Theta1.transpose())
    a2= 1/(1+np.exp(-z2))
    one_matrix= np.ones((m,1))
    a2= np.append(one_matrix, a2, axis=1)
    z3= np.dot(a2, Theta2.transpose())
    a3= 1/(1+np.exp(-z3))
    y_vect= np.zeros((m,10))
    for i in range (m):
        y_vect[i, int(Y[i])]= 1
    J= (1/m)*(np.sum(np.sum(-y_vect * np.log(a3)-(1-y_vect)*np.log(1-a3))))+(lamb/(2*m))*(sum(sum(pow(Theta1[:,1:], 2)))+sum(sum(pow(Theta2[:,1:], 2))))
    delta3= a3- y_vect
    delta2= np.dot(delta3, Theta2)*a2*(1-a2)
    delta2= delta2[:,1:]
    Theta1[:,0]= 0
    Theta1_grad= (1/m)*np.dot(delta2.transpose(),a1) + (lamb/m)*Theta1
    Theta2[:,0]= 0
    Theta2_grad= (1/m)*np.dot(delta3.transpose(),a2) + (lamb/m)*Theta2
    grad= np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return J, grad