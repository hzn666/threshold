import pandas as pd

data = pd.read_csv('data/ipinyou/1458/train.bid.lin.csv')
data[data['day']==12].to_csv('data/ipinyou/1458/12.csv', index=None)

data = pd.read_csv('data/ipinyou/1458/test.bid.lin.csv')
data[data['day']==13].to_csv('data/ipinyou/1458/13.csv', index=None)