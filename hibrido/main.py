from ga import *
from perceptron import *
import time
import matplotlib.pyplot as plt

epocasPlot = []
timePlot = []
clf = Perceptron()

def main ():
  ini = time.time()
  initial_population = get_initial_population(clf.X)
  population = initial_population
  epocas = 0  
  while True:
    #ini = time.time()
    scores = score(clf, population)
    scores.sort(reverse=True)
    if scores[0] == 768 or epocas == 1000:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      return population
      break
    population,scores = selectionFirst(population,scores )
    new_population = crossover([], population)
    new_population = mutation(new_population, clf.X)
    new_scores = score(clf, new_population)
    population,scores = selection(population, scores, new_population, new_scores)
    #print(scores)
    if new_scores[0] == 768 or epocas == 1000:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      return population
      break
    epocas+=1
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