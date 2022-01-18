# Alunos
# Guilherme Volney Mota Amaral
# João Victor Ferro
# Arthur Bernardo Sávio de Melo
# Problema: separar duas classes de flores Iris (Setosa e Versicolor)

from ga import *
from adaline import *

clf = Adaline()

initial_population = get_initial_population(clf.X)
population = initial_population

while True:
  scores = score(clf, population)
  scores.sort(reverse=True)
  if scores[0] == 768:
    print('Pesos: ', population[0])
    break
  new_population = crossover([], population)
  new_population = mutation(new_population, clf.X)
  new_scores = score(clf, new_population)
  population = selection(population, scores, new_population, new_scores)
  print(scores)
  if new_scores[0] == 768:
    print('Pesos: ', population[0])
    break