import pandas as pd

df = pd.read_csv('diabetesDataset.csv')
df.head()

testOne = df.query('Outcome == 1')[:134]
trainOne = df.query('Outcome == 1')[134:]

testZero = df.query('Outcome == 0')[:250]
trainZero = df.query('Outcome == 0')[250:]

dataSetTest = pd.concat([testOne, testZero])
dataSetTraint = pd.concat([trainOne, trainZero])

dataSetTest.to_csv("dataSetTest.csv")
dataSetTraint.to_csv("dataSetTraint.csv")

print(dataSetTest)
print('----------')
print(dataSetTraint)
