import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

from src.util.plotter import Plotter
import matplotlib.pyplot as plt
from src.config.config import Config
import json
import numpy as np


def plot_all_res(log_list, name_list, title=' '):
    Plotter.plot_many_target_agent_reward(path_list=log_list,
                                          name_list=name_list,
                                          title='Case 1')


def plot_q_table(file_name, index, save_path):
    index_str = str(index[0]) + '_' + str(index[1])
    table_list = [0 for i in range(10000)]
    count = 0
    plotter = Plotter(log_path=CURRENT_PATH)
    with open(file=file_name) as f:
        res = json.load(fp=f)
        for sample in res:
            table_list[sample['INDEX']] = np.array(sample['Q_TABLE'])
            count += 1
        merge_q_list = []
        for i in range(4):
            q_list = []
            for j in range(count):
                q_list.append(np.mean(table_list[j][i][index]))
            merge_q_list.append(q_list)
    #
    # plotter.plot_fig(fig_num=10,
    #                  x=[0 for _ in range(len(q_list))],
    #                  y=q_list,
    #                  title='Q table on ' + index_str,
    #                  x_lable='Episode',
    #                  y_label='Reward',
    #                  col_id=1,
    #                  marker=Plotter.markers[0]
    #                  )

    # self.action_dict = ['S', 'N', 'W', 'E']
    label_list = ('Down', 'Up', 'Left', 'Right')
    for i in range(4):
        a = Plotter(log_path='')
        a.plot_fig(fig_num=4, col_id=i, x=[i for i in range(len(merge_q_list[i]))], y=merge_q_list[i],
                   title='Q Value on ' + '(' + index_str + ')',
                   x_lable='Episode', y_label='Q value',
                   marker=Plotter.markers[i],
                   label=label_list[i]
                   )
    plt.savefig(save_path + '/' + index_str + '.pdf', format='pdf', bbox_inces='tight')
    plt.show()


if __name__ == '__main__':
    Plotter.plot_many_target_agent_reward(path_list=[[
        "/Users/Luke/Documents/RoutePlanningWithReinforcementLearning/log/simpleGridWorld/2018-06-03_21-08-12_case3"]],
        name_list=['case3'],
        title='case3')
    # plot_q_table(file_name='/Users/Luke/Documents/RoutePlanningWithReinforcementLearning/log/simpleGridWorld/2018-06-03_21-00-40_case2/loss/TargetAgent_test_.log',
    #              index=(0, 3),
    #              save_path='/Users/Luke/Documents/RoutePlanningWithReinforcementLearning/log/simpleGridWorld/2018-06-03_21-00-40_case2')
    #
