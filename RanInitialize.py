import numpy as np
def intialise(a,b):
    epsilon= 0.15
    C= np.random.rand(a,b+1)*(2*epsilon) - epsilon
    return C
