import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.colors import ListedColormap
class Perceptron():
  def __init__(self, eta:float=None, epochs: int= None):
    self.weights= np.random.randn(3)*1e-4
    print(self.weights)
    training= eta is not None and epochs is not None
    if training:
      print('Initial weight', self.weights)
    self.eta= eta
    self.epochs= epochs
  def _z_outcome(self, inputs, weights):
    return np.dot(inputs, weights)
  def activation_function(self,z):
    return np.where(z>0, 1,0)
  def fit(self, X,y):
    self.X= X
    self.y= y
    X_with_bias= np.c_[self.X,-np.ones((len(self.X),1))]
    print('X with bias is', X_with_bias)
    for epoch in range(self.epochs):
      z= self._z_outcome(X_with_bias, self.weights)
      y_hat= self.activation_function(z)
      self.error= self.y-y_hat
      print('The error is', self.error)
      self.weights= self.weights+self.eta*np.dot(X_with_bias.T, self.error)
  def total_loss(self):
    total_loss= np.sum(self.error)
    print('Total loss is', total_loss)
  def _create_dir_return_path(self, model_dir, file_name):
      os.makedirs(model_dir, exist_ok= True)
      return os.path.join(model_dir, file_name)
  def save(self,file_name, model_dir=None):
      if model_dir is not None:
          model_file_path= self._create_dir_return_path(model_dir,file_name)
          joblib.dump(self, model_file_path)
      else:
          model_file_path= self._create_dir_return_path("Model", file_name)
          joblib.dump(self, model_file_path)
  def load(self, file_path):
    return joblib.load(file_path)
# XOR= {
#     "X1":[0,0,1,1],
#     "X2":[0,1,0,1],
#     "Y":[0,1,1,0]
# }
# df_XOR= pd.DataFrame(XOR)
# def prepare_data(df, target_col= "Y"):
#   X= df.drop(target_col, axis=1)
#   Y= df[target_col]
#   return X, Y
# X, Y= prepare_data(df_XOR)
# print(X)
# print(Y)
# eta= 0.1
# epochs= 15
# model_XOR= Perceptron(eta=eta, epochs= epochs)
# model_XOR.fit(X, Y)
# print(model_XOR.total_loss())
AND= {
    "X1":[0,0,1,1],
    "X2":[0,1,0,1],
    "Y":[0,0,0,1]
}
df_AND= pd.DataFrame(AND)
print(df_AND)
def prepare_data(df, target_call= "Y"):
   X= df.drop(target_call, axis=1)
   Y= df[target_call]
   return(X,Y)
X,Y= prepare_data(df_AND)
print(X)
print(Y)
eta=0.1
epochs= 15
model_AND= Perceptron(eta= eta, epochs= epochs)
model_AND.fit(X,Y)
print(model_AND.total_loss())
def save_plot(df,model, file_name= "plot.png", plot_dir= "plots"):
  def _create_base_plot(df):
    df.plot(kind= "scatter", x= "X1", y= "X2", S= 100, cmap= "coolwarm")
    plt.axhline(y=0,color="black", linestyle="--", linewidth=1 )
    plt.axvline(y=1, color="black", linestyle="--", linewidth=1)
    figure=plt.gcf()
    figure.set_size_inches(10,8)
        
