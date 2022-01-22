import numpy as np
import time

class Adaline(object):
    def __init__(self, learningRate=0.02, epochs=1000):
      self.learningRate = learningRate
      self.epochs = epochs
      self.epocasPlot = []
      self.timePlot = []
      
    def train(self, X, y):
      ini = time.time()
      self._weights = np.zeros(X.shape[1] + 1)
      self.errors = []
      
      for _ in range (self.epochs):
        #ini = time.time()
        errors = 0
        for xi, target in zip(X, y):
          error = (target - self.predict(xi))
          errors += error
                    
          update = self.learningRate * error
          self._weights[1:] += update * xi
          self._weights[0] += update
          
        self.errors.append(errors)
        fim = time.time()
        self.timePlot.append(fim-ini)
        print(f'Tempo {self.timePlot}')
        self.epocasPlot.append(len(self.epocasPlot)+1)
      #return self
    
    def net_input(self, X):
      bias = self._weights[0]
      return np.dot(X, self._weights[1:]) + bias
    def activation_function(self, X):
      return self.net_input(X)
    def predict(self, X):
      return np.where(self.activation_function(X) >= 0.0, 1, 0)
