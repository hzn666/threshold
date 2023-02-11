import pandas as pd

data = pd.read_csv('data/ipinyou/1458/15.csv')

# data_clk = data[data['clk'] == 1]
# data_sorted = data_clk.sort_values(by='pctr').to_csv('data/ipinyou/1458/15_clk.csv', index=None)

data_useless = data[data['pctr'] < 0.0001159330495283939]
print(sum(data_useless['market_price']))
