import numpy as np
import time
import math

class Adaline(object):
    def __init__(self, learningRate=0.4, epochs=1000):
      self.learningRate = learningRate
      self.epochs = epochs
      self.epocasPlot = []
      self.timePlot = []
      self.acertosApurado = 0
      self.errosApurado = 0
      self.u = 0
    def train(self, X, y):
      ini = time.time()
      self._weights = np.zeros(1 + X.shape[1])
      self.errors = []
      
      for _ in range (self.epochs):
        #ini = time.time()
        errors = 0
        for xi, target in zip(X, y):
          error = (target - self.predict(xi))
          errors += int(error != 0.0)
                    
          update = self.learningRate * error
          self._weights[1:] += update * xi
          self._weights[0] += update
        self.errors.append(errors)
        
        fim = time.time()
        self.timePlot.append(fim-ini)
        #print(f'Tempo {self.timePlot}')
        self.epocasPlot.append(len(self.epocasPlot)+1)
      return self
    
    def net_input(self, X):
      bias = self._weights[0]
      output = 0
      i = 0
      for i in range(len(X)):
        output = X[i] * self._weights[i+1]
      output = bias + output
      return output
    
    def activation_function(self, X):
      return self.net_input(X)

    def predict(self, X):
      return np.where(self.activation_function(X) >= 0.0, 1, 0)
    
    def test(self, X, y):      
      for xi, target in zip(X, y):
        i = 0
        output = 0
        bias = self._weights[0]
        for i in range(len(xi)):
          output = xi[i] * self._weights[i+1]
        self.u = output + bias
        self.saida(self.u, target)
        self.u = 0
    
    def saida(self, u, target):
      newTarget = -1
      if u >= 0.0:
        newTarget = 1
        if newTarget == target:
          self.acertosApurado+=1
          return
        else:
          self.errosApurado+=1
      else:
        newTarget = 0
        if newTarget == target:
          self.acertosApurado+=1
          return
        else:
          self.errosApurado+=1