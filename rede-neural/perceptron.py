import numpy as np
import time
import math

class Perceptron(object):
    def __init__(self, learningRate=0.2, epochs=1600):
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
      output = bias
      return output
    
    def activation_function(self, X):
      x = self.net_input(X)
      if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
      else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig
    def predict(self, X):
      return self.activation_function(X)
    
    def test(self, X, y):      
      for xi, target in zip(X, y):
        i = 0
        output = 0
        bias = self._weights[0]
        for i in range(len(xi)):
          output = xi[i] * self._weights[i+1]
        self.u = output + bias
        print(f'Valor de u - {self.u}, TARGET - {target}')
        self.saida(self.u, target)
        self.u = 0
    
    def saida(self, u, target):
      a = 1 / (1 + math.exp(-u))
      print(f'sub {abs(target - a)}')
      if abs(target - a) == 0 or abs(target - a) == 1:
        self.acertosApurado+=1
      else:
        self.errosApurado+=1