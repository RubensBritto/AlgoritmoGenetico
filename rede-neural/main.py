import pandas as pd
from perceptron import *
import time
import matplotlib.pyplot as plt


clf = Perceptron(epochs=5000)

def train():
  df = pd.read_csv('dataSetTrain2.csv')
  df.head()

  X = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
  y = df.iloc[0:,8].values
  
  clf.train(X, y)
  print(f'Pesos calculados: {clf._weights[1:]}')

def plot():
  plt.plot(clf.epocasPlot,clf.timePlot)
  plt.ylabel('Time(s)')
  plt.xlabel('Epocas')
  plt.show()
  
def test():
  df = pd.read_csv('dataSetTest2.csv')
  df.head()

  X_2 = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
  y_2 = df.iloc[0:,8].values
  clf.test(X_2, y_2)
  
  print(f'Acertos: {clf.acertosApurado}')
  print(f'Erros: {clf.errosApurado}')
  print(f'Acuracia: {clf.acertosApurado/(clf.acertosApurado+clf.errosApurado)}')
  print(f'Pesos calculados: {clf._weights[1:]}')

if __name__ == "__main__":
  train()
  test()
  #plot()