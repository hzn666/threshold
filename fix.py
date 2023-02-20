import pandas as pd


def fix_threshold_bid(data, threshold, para):
    clk = 0
    imps = 0
    pctr = 0
    spend = 0
    budget = data['market_price'].sum() / para
    for row in data.itertuples(index=False):
        if budget > 0:
            if row[1] >= threshold and budget - row[2] >= 0:
                clk += row[0]
                imps += 1
                pctr += row[1]
                spend += row[2]
                budget -= row[2]

    return clk, pctr, imps, spend


def init_threshold(data, para):
    """
    获取初始阈值
    :param para: 预算限制
    :return: 初始阈值
    """
    budget = data['market_price'].sum() / para
    data_sorted = data.sort_values(by='pctr', ascending=False)
    for row in data_sorted.itertuples(index=False):
        if budget > 0:
            budget -= row[2]
            if budget < 0:
                return row[1]


if __name__ == '__main__':
    train_data = pd.read_csv('data/ipinyou/1458/12.csv')
    test_data = pd.read_csv('data/ipinyou/1458/13.csv')

    for para in [2, 4, 8, 16]:
        threshold = init_threshold(train_data, para)
        print(fix_threshold_bid(test_data, threshold, para))
