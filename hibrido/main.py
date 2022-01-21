from ga import *
from adaline import *

epocas = 0
clf = Adaline()
initial_population = get_initial_population(clf.X)
population = initial_population

while epocas <=10000:
  print(f'epocas - {epocas}')
  scores = score(clf, population)
  scores.sort(reverse=True)
  if scores[0] == 768:
    print('Pesos: ', population[0])
    break
  population,scores = selectionFirst(population,scores )
  new_population = crossover([], population)
  new_population = mutation(new_population, clf.X)
  new_scores = score(clf, new_population)
  population,scores = selection(population, scores, new_population, new_scores)
  print(scores)
  if new_scores[0] == 768:
    print('Pesos: ', population[0])
    break
  epocas+=1