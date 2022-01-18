import numpy as np
import pandas as pd

class Adaline(object):
    def __init__(self):
      self.df = pd.read_csv('diabetesDataset.csv')
      self.df.head()

      self.X = self.df.iloc[0:,[0,1,2,3,4,5,6,7]].values
      self.y = self.df.iloc[0:,8].values

    def net_input(self, individual, weight_):
      #print(individual)
      #print(weight_)
      return np.dot(individual, weight_[1:]) + weight_[0]
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