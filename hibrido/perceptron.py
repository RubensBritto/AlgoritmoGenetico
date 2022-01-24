import numpy as np
import pandas as pd
import math

class Perceptron(object):
    def __init__(self):
      self.df = pd.read_csv('dataSetTrain2.csv')
      self.df.head()
      self.acertosApurado = 0
      self.errosApurado = 0

      self.X = self.df.iloc[0:,[0,1,2,3,4,5,6,7]].values
      self.y = self.df.iloc[0:,8].values

    def net_input(self, individual, weight_):
      bias = weight_[0]
      output = 0
      i = 0
      for i in range(len(individual)):
        output = individual[i] * weight_[i+1]
      output = bias + output
      return output
    
    def activation_function(self, individual, weight_):
      x = self.net_input(individual,weight_ )
      if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
      else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig
    def predict(self, weight_):
      score = 0
      for _, data in self.df.iterrows():
        df_individual = []
        answer = None
        for i, v in data.items():
          if i != "Outcome":
            df_individual.append(v)
          else:
            answer = v    
        if len(df_individual) != 8 or len(weight_) != 9:
          break
        prediction = self.activation_function(df_individual, weight_)
        if prediction == answer:
          score += 1
      return score
    
    def test(self, X, y, weights):      
      for xi, target in zip(X, y):
        i = 0
        output = 0
        bias = weights[0]
        for i in range(len(xi)):
          output = xi[i] * weights[i+1]
        self.u = output + bias
        self.saida(self.u, target)
        self.u = 0
     
    def saida(self, u, target):
      a = 1 / (1 + math.exp(-u))
      print(f'sub {abs(target - a)}')
      if abs(target - a) == 0 or abs(target - a) == 1:
        self.acertosApurado+=1
      else:
        self.errosApurado+=1
