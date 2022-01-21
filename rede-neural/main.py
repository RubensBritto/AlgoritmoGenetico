import pandas as pd
from adaline import *
import time

clf = Adaline(epochs=100)

def main():
  df = pd.read_csv('diabetesDataset.csv')
  df.head()

  X = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
  y = df.iloc[0:,8].values
  
  clf.train(X, y)
  print(f'Pesos calculados: {clf._weights}')

if __name__ == "__main__":
  ini = time.time()
  main()
  fim = time.time()
  print(f'Tempo execução: {fim-ini}') 