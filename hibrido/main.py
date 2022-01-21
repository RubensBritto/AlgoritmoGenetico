from ga import *
from adaline import *
import time


def main ():
  clf = Adaline()
  initial_population = get_initial_population(clf.X)
  population = initial_population
  epocas = 0  
  while True:
    scores = score(clf, population)
    scores.sort(reverse=True)
    if scores[0] == 768 or epocas == 10:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      break
    population,scores = selectionFirst(population,scores )
    new_population = crossover([], population)
    new_population = mutation(new_population, clf.X)
    new_scores = score(clf, new_population)
    population,scores = selection(population, scores, new_population, new_scores)
    #print(scores)
    if new_scores[0] == 768 or epocas == 10:
      print('Pesos: ', population[0])
      print(f'epocas - {epocas}')
      break
    epocas+=1

if __name__ == "__main__":
  ini = time.time()
  main()
  fim = time.time()
  print(f'Tempo execução: {fim-ini}') 