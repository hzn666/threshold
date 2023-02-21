import argparse
import os
import time
from collections import deque
from ppo import *
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def stack_frames(stacked_frames, frame, is_new_episode=False):
    if is_new_episode:
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    state = np.stack(stacked_frames, axis=0).reshape(-1)
    return state, stacked_frames


def get_budget(data):
    _ = []
    for day in data['day'].unique():
        current_day_budget = sum(data[data['day'].isin([day])]['market_price'])
        _.append(current_day_budget)

    return _


def reward_func(reward_type, result):
    if reward_type == 'pctr':
        return result['win_pctr']
    else:
        return result['win_clks'] / 1000


def bid(data, budget, **cfg):
    bid_imps = 0
    bid_clks = 0
    bid_pctr = 0
    win_imps = 0
    win_clks = 0
    win_pctr = 0
    spend = 0
    action = []
    bid_action = []

    if len(data) == 0:
        return {
            'bid_imps': bid_imps,
            'bid_clks': bid_clks,
            'bid_pctr': bid_pctr,
            'win_imps': win_imps,
            'win_clks': win_clks,
            'win_pctr': win_pctr,
            'spend': spend,
            'bid_action': bid_action
        }

    for row in data.itertuples(index=False):
        if budget > 0:
            if row[1] >= cfg['threshold'] and budget - row[2] >= 0:
                win_clks += row[0]
                win_imps += 1
                win_pctr += row[1]
                spend += row[2]
                budget -= row[2]
                action.append(1)
            else:
                action.append(0)

            bid_imps += 1
            bid_clks += row[0]
            bid_pctr += row[1]
        else:
            action.append(0)

    # print(len(action), len(data))
    data['bid'] = action
    bid_action.extend(data.values.tolist())

    return {
        'bid_imps': bid_imps,
        'bid_clks': bid_clks,
        'bid_pctr': bid_pctr,
        'win_imps': win_imps,
        'win_clks': win_clks,
        'win_pctr': win_pctr,
        'spend': spend,
        'bid_action': bid_action
    }


def rtb(data, init_state, budget_para, RL, config, train=True):
    if train:
        RL.is_test = False
    else:
        RL.is_test = True

    time_fraction = config['time_fraction']

    budget = get_budget(data)[0] / budget_para
    day_budget = [budget]

    day_bid_imps = []
    day_bid_clks = []
    day_bid_pctr = []
    day_win_imps = []
    day_win_clks = []
    day_win_pctr = []
    day_spend = []
    day_bid_action = []

    day_action = []
    day_reward = []

    stacked_frames = deque([np.zeros(8) for _ in range(4)], maxlen=4)

    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    init_threshold = {
        2: 0.0003140721528325,
        4: 0.0005973504739813,
        8: 0.0009133457788266,
        16: 0.0012582762865349
    }

    for slot in range(0, time_fraction):
        if slot == 0:
            frame = [0, 0, 0, 0] + init_state
            state, stacked_frames = stack_frames(stacked_frames, frame, is_new_episode=True)
            threshold = init_threshold[budget_para]
        else:
            state = next_state

        action = RL.take_action(state)
        threshold = threshold * (1 + action[0])
        # day_action.append(action[0])
        day_action.append(threshold)

        slot_data = data[data['time_fraction'] == slot]
        slot_result = bid(slot_data, day_budget[-1], threshold=threshold)

        slot_reward = reward_func(config['reward_type'], slot_result)
        day_reward.append(slot_reward)

        day_bid_imps.append(slot_result['bid_imps'])
        day_bid_clks.append(slot_result['bid_clks'])
        day_bid_pctr.append(slot_result['bid_pctr'])
        day_win_imps.append(slot_result['win_imps'])
        day_win_clks.append(slot_result['win_clks'])
        day_win_pctr.append(slot_result['win_pctr'])
        day_spend.append(slot_result['spend'])
        day_budget.append(day_budget[-1] - slot_result['spend'])
        day_bid_action.extend(slot_result['bid_action'])

        if slot == time_fraction - 1:
            done = 1
            day_budget.pop(-1)
        else:
            done = 0

        left_slot_ratio = (time_fraction - 2 - slot) / (time_fraction - 1)
        slot_avg_pctr = slot_data['pctr'].mean() if len(slot_data) else 0
        slot_std_pctr = slot_data['pctr'].std() if len(slot_data) else 0
        slot_q1_pctr = slot_data['pctr'].quantile(0.25) if len(slot_data) else 0
        slot_q3_pctr = slot_data['pctr'].quantile(0.75) if len(slot_data) else 0
        next_frame = [
            (day_budget[-1] / day_budget[0]) / left_slot_ratio if left_slot_ratio else day_budget[-1] / day_budget[0],
            day_spend[-1] / day_budget[0],
            day_win_clks[-1] / day_win_imps[-1] if day_win_imps[-1] else 0,
            day_win_imps[-1] / day_bid_imps[-1] if day_bid_imps[-1] else 0,
            slot_avg_pctr,
            slot_std_pctr,
            slot_q1_pctr,
            slot_q3_pctr
        ]
        next_state, stacked_frames = stack_frames(stacked_frames, next_frame)

        if train and not done:
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(slot_reward)
            transition_dict['dones'].append(done)

        if done:
            break

    if train:
        actor_loss, critic_loss = RL.update(transition_dict)
        global actor_loss_cnt
        actor_loss_cnt += 1
        RL.writer.add_scalar('actor_loss', actor_loss, actor_loss_cnt)
        global critic_loss_cnt
        critic_loss_cnt += 1
        RL.writer.add_scalar('critic_loss', critic_loss, critic_loss_cnt)

    if train:
        result = "训练"
    else:
        result = "测试"

    print(
        result + "：点击数 {}, 真实点击数 {}, pCTR {:.4f}, 真实pCTR {:.4f}, 赢标数 {}, 真实曝光数 {}, 花费 {}, CPM {:.4f}, CPC {:.4f}, 奖励 {:.2f}".format(
            int(sum(day_win_clks)),
            int(sum(day_bid_clks)),
            sum(day_win_pctr),
            sum(day_bid_pctr),
            sum(day_win_imps),
            sum(day_bid_imps),
            sum(day_spend),
            sum(day_spend) / sum(day_win_imps),
            sum(day_spend) / sum(day_win_clks),
            sum(day_reward)
        )
    )

    if train:
        global train_reward_cnt
        train_reward_cnt += 1
        RL.writer.add_scalar('train_reward', sum(day_reward), train_reward_cnt)
    else:
        global test_reward_cnt
        test_reward_cnt += 1
        RL.writer.add_scalar('test_reward', sum(day_reward), test_reward_cnt)

    episode_record = [
        int(sum(day_win_clks)),
        int(sum(day_bid_clks)),
        sum(day_win_pctr),
        sum(day_bid_pctr),
        sum(day_win_imps),
        sum(day_bid_imps),
        sum(day_spend),
        sum(day_spend) / sum(day_win_imps),
        sum(day_spend) / sum(day_win_clks),
        sum(day_reward)
    ]
    return episode_record, day_action, day_bid_action


def main(budget_para, RL, config):
    record_path = os.path.join(config['result_path'], config['campaign_id'])
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    train_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], '12.csv'))
    test_data = pd.read_csv(os.path.join(config['data_path'], config['campaign_id'], '13.csv'))

    header = ['clk', 'pctr', 'market_price', 'day']

    if config['time_fraction'] == 96:
        header.append('96_time_fraction')
    elif config['time_fraction'] == 48:
        header.append('48_time_fraction')
    elif config['time_fraction'] == 24:
        header.append('24_time_fraction')

    train_data = train_data[header]
    train_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']
    test_data = test_data[header]
    test_data.columns = ['clk', 'pctr', 'market_price', 'day', 'time_fraction']

    epoch_train_record = []
    epoch_train_action = []

    epoch_test_record = []
    epoch_test_action = []

    avg_pctr = train_data['pctr'].mean()
    std_pctr = train_data['pctr'].std()
    q1 = train_data['pctr'].quantile(0.25)
    q3 = train_data['pctr'].quantile(0.75)
    init_state = [avg_pctr, std_pctr, q1, q3]

    for epoch in range(config['train_epochs']):
        print('第{}轮'.format(epoch + 1))
        train_record, train_action, train_bid_action = rtb(train_data, init_state, budget_para, RL, config)
        test_record, test_action, test_bid_action = rtb(test_data, init_state, budget_para, RL, config, train=False)

        epoch_train_record.append(train_record)
        epoch_train_action.append(train_action)

        epoch_test_record.append(test_record)
        epoch_test_action.append(test_action)

        if config['save_bid_action']:
            bid_action_path = os.path.join(record_path, 'bid_action')
            if not os.path.exists(bid_action_path):
                os.makedirs(bid_action_path)

            train_bid_action_df = pd.DataFrame(data=train_bid_action,
                                               columns=['clk', 'pctr', 'market_price', 'day', 'time_fraction',
                                                        'bid_price', 'win'])
            train_bid_action_df.to_csv(bid_action_path + '/train_' + str(budget_para) + '_' + str(epoch) + '.csv',
                                       index=False)

            test_bid_action_df = pd.DataFrame(data=test_bid_action,
                                              columns=['clk', 'pctr', 'market_price', 'day', 'time_fraction',
                                                       'bid_price', 'win'])
            test_bid_action_df.to_csv(bid_action_path + '/test_' + str(budget_para) + '_' + str(epoch) + '.csv',
                                      index=False)

    columns = ['clks', 'real_clks', 'pctr', 'real_pctr', 'imps', 'real_imps', 'spend', 'CPM', 'CPC', 'reward']

    train_record_df = pd.DataFrame(data=epoch_train_record, columns=columns)
    train_record_df.to_csv(record_path + '/train_episode_results_' + str(budget_para) + '.csv')

    train_action_df = pd.DataFrame(data=epoch_train_action)
    train_action_df.to_csv(record_path + '/train_episode_actions_' + str(budget_para) + '.csv')

    test_record_df = pd.DataFrame(data=epoch_test_record, columns=columns)
    test_record_df.to_csv(record_path + '/test_episode_results_' + str(budget_para) + '.csv')

    test_action_df = pd.DataFrame(data=epoch_test_action)
    test_action_df.to_csv(record_path + '/test_episode_actions_' + str(budget_para) + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ipinyou')
    parser.add_argument('--campaign_id', type=str, default='1458')
    parser.add_argument('--result_path', type=str, default='result')
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--feature_num', type=int, default=32)
    parser.add_argument('--action_num', type=int, default=1)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=5e-3)
    parser.add_argument('--lmbda', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--budget_para', nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--train_epochs', type=int, default=1000)
    parser.add_argument('--save_bid_action', type=bool, default=False)
    parser.add_argument('--reward_type', type=str, default='clk', help='op, nop_2.0, clk')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    config = vars(args)

    str_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    config['result_path'] = config['result_path'] + '-camp={}-seed={}-{}'.format(config['campaign_id'], config['seed'],
                                                                                 str_time)
    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])

    budget_para_list = list(map(int, config['budget_para']))

    actor_loss_cnt = 0
    critic_loss_cnt = 0
    train_reward_cnt = 0
    test_reward_cnt = 0

    for i in budget_para_list:
        RL = PPOContinuous(
            config['feature_num'],
            config['action_num'],
            config['actor_lr'],
            config['critic_lr'],
            config['lmbda'],
            config['eps'],
            config['gamma'],
            config['epochs'],
            config['campaign_id'],
            i,
            config['seed'],
            str_time
        )

        print('当前预算条件{}'.format(i))
        main(i, RL, config)
