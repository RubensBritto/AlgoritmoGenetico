import pandas as pd

df = pd.read_csv('diabetesDataset.csv')
#df.head()

#coloca a coluna Outcome separada e depois apago do vetor
saida= df.iloc[:,[8]]
del df['Outcome']

def minmax_norm(df_input):
    return (df - df.min()) / ( df.max() - df.min())

df = minmax_norm(df)
df['Outcome'] = saida
#print(df)

testOne = df.query('Outcome == 1')[:134]
trainOne = df.query('Outcome == 1')[134:]

testZero = df.query('Outcome == 0')[:250]
trainZero = df.query('Outcome == 0')[250:]

dataSetTest = pd.concat([testOne, testZero])
dataSetTraint = pd.concat([trainOne, trainZero])


dataSetTest.to_csv("dataSetTest2.csv", encoding='utf-8', index=False)
dataSetTraint.to_csv("dataSetTrain2.csv", encoding='utf-8', index=False)
