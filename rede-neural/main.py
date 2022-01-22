import pandas as pd
from adaline import *
import time
import matplotlib.pyplot as plt


clf = Adaline(epochs=10)

def main():
  df = pd.read_csv('diabetesDataset.csv')
  df.head()

  X = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
  y = df.iloc[0:,8].values
  
  clf.train(X, y)
  print(f'Pesos calculados: {clf._weights}')

def plot():
  plt.plot(clf.epocasPlot,clf.timePlot)
  plt.ylabel('Time(s)')
  plt.xlabel('Epocas')
  plt.show()

if __name__ == "__main__":
  ini = time.time()
  main()
  plot()
  fim = time.time()
  print(f'Tempo execução: {fim-ini}') 