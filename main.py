from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RanInitialize import intialise
from Prediction import predict
from scipy.optimize import minimize 
data= loadmat("mnist-original.mat")
X= data["data"]
X= X.transpose()
X= X/255
Y= data["label"]
print(Y)
Y= Y.flatten()
print(Y)
X_train= X[:60000,:]
Y_train= Y[:60000]
X_test= X[60000:,:]
Y_test= Y[60000:]
m= X.shape[0]
print(m)
input_layer_size= 784
hidden_layer_size= 100
num_labels= 10
initial_theta1= intialise(hidden_layer_size, input_layer_size)
initial_theta2= intialise(num_labels, hidden_layer_size)
initial_nn_parms= np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()))
max_iter= 100
lambda_reg= 0.1
my_args= (input_layer_size, hidden_layer_size, num_labels, X_train, Y_train, lambda_reg)
results= minimize(neural_network, x0= initial_nn_parms, args= my_args, options= {'disp': True, 'maxiter': max_iter}, method= 'L-BFGS-B', jac= True)
nn_parms= results["x"]
Theta1= np.reshape(nn_parms[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2= np.reshape(nn_parms[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
pred= predict(Theta1, Theta2, X_test)
print("Test set accuracy", (np.mean(pred==Y_test)*100))
pred= predict(Theta1, Theta2, X_train)
print("Training set accuracy", (np.mean(pred== Y_train)*100))
true_positive= 0 
for i in range(len(pred)):
    if pred[i]== Y_train[i]:
        true_positive+=1
false_positive= len(Y_train)- true_positive
print("precision= ", true_positive/(true_positive+false_positive))
np.savetxt("Theta1.txt", Theta1, delimiter= ' ')
np.savetxt("Theta2.txt", Theta1, delimiter= ' ')