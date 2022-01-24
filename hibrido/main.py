from ga import *
from perceptron import *
import time
import matplotlib.pyplot as plt
import pandas as pd

epocasPlot = []
timePlot = []
clf = Perceptron()

df = pd.read_csv('dataSetTrain2.csv')
df.head()

X_train = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
y_train = df.iloc[0:,8].values


def train(population):
  scores = []
  for individual in population:
    score = clf.train(X_train, y_train, individual)
    #if score == 0:
      #print('Score 0')
    scores.append(score)
  return scores

def main ():
  ini = time.time()
  initial_population = get_initial_population(X_train)
  population = initial_population
  epocas = 0  
  while True:
    #ini = time.time()
    scores = train(population)
    scores.sort(reverse=True)
    if scores[0] == 0 or epocas == 20:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      print(f'score: {scores}')
      
      return population
    population,scores = selectionFirst(population,scores )
    new_population = crossover([], population)
    new_population = mutation(new_population, X_train)
    new_scores = train(new_population)
    population,scores = selection(population, scores, new_population, new_scores)
    #print(scores)
    if new_scores[0] == 0 or epocas == 20:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      print(f'score: {new_scores}')
      return population
    epocas+=1
    new_population.clear
    new_scores.clear
    fim = time.time()
    timePlot.append(fim-ini)
    #print(f'timePlot {timePlot}')
    epocasPlot.append(epocas)

def plot():
  plt.plot(epocasPlot,timePlot)
  plt.ylabel('Time(s)')
  plt.xlabel('Epocas')
  plt.show()

def test(population):
  df = pd.read_csv('dataSetTest2.csv')
  df.head()

  X_2 = df.iloc[0:,[0,1,2,3,4,5,6,7]].values
  y_2 = df.iloc[0:,8].values
  clf.test(X_2,y_2,population[0])
  
  print(f'Acertos: {clf.acertosApurado}')
  print(f'Erros: {clf.errosApurado}')
  print(f'Acuracia: {clf.acertosApurado/(clf.acertosApurado+clf.errosApurado)}')

if __name__ == "__main__":
  pop = main()
  test(pop)
  #plot()