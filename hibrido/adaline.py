import numpy as np
import pandas as pd

class Adaline(object):
    def __init__(self):
      self.df = pd.read_csv('dataSetTrain.csv')
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
      return self.net_input(individual, weight_)
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
        prediction = np.where(self.activation_function(df_individual, weight_) >= 0.0, 1, 0)
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
        newTarget = -1
        if u>=0.0:
          newTarget = 1
          if newTarget == target:
            self.acertosApurado+=1
          else:
            self.errosApurado+=1
        else:
          newTarget = 0
          if newTarget == target:
            self.acertosApurado+=1
          else:
            self.errosApurado+=1
