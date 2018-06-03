import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import math as M
from scipy.interpolate import interp1d
from itertools import groupby
import seaborn as sns
import os
import glob

sns.set_style('ticks')


class Plotter(object):
    markers = ('+', 'x', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen', 'darkorange', 'oldlace', 'chocolate',
                  'purple', 'lightskyblue', 'gray', 'seagreen', 'antiquewhite',
                  'snow', 'darkviolet', 'brown', 'skyblue', 'mediumaquamarine', 'midnightblue', 'darkturquoise',
                  'sienna', 'lightsteelblue', 'gold', 'teal', 'blueviolet', 'mistyrose', 'seashell', 'goldenrod',
                  'forestgreen', 'aquamarine', 'linen', 'deeppink', 'darkslategray', 'mediumseagreen', 'dimgray',
                  'mediumpurple', 'lightgray', 'khaki', 'dodgerblue', 'papayawhip', 'salmon', 'floralwhite',
                  'lightpink', 'gainsboro', 'coral', 'indigo', 'darksalmon', 'royalblue', 'navy', 'orangered',
                  'cadetblue', 'orchid', 'palegreen', 'magenta', 'honeydew', 'darkgray', 'palegoldenrod', 'springgreen',
                  'lawngreen', 'palevioletred', 'olive', 'red', 'lime', 'yellowgreen', 'aliceblue', 'orange',
                  'chartreuse', 'lavender', 'paleturquoise', 'blue', 'azure', 'yellow', 'aqua', 'mediumspringgreen',
                  'cornsilk', 'lightblue', 'steelblue', 'violet', 'sandybrown', 'wheat', 'greenyellow', 'darkred',
                  'mediumslateblue', 'lightseagreen', 'darkblue', 'moccasin', 'lightyellow', 'turquoise', 'tan',
                  'mediumvioletred', 'mediumturquoise', 'limegreen', 'slategray', 'lightslategray', 'mintcream',
                  'darkgreen', 'white', 'mediumorchid', 'firebrick', 'bisque', 'darkcyan', 'ghostwhite', 'powderblue',
                  'tomato', 'lavenderblush', 'darkorchid', 'cornflowerblue', 'plum', 'ivory', 'darkgoldenrod', 'green',
                  'burlywood', 'hotpink', 'cyan', 'silver', 'peru', 'thistle', 'indianred', 'olivedrab',
                  'lightgoldenrodyellow', 'maroon', 'black', 'crimson', 'darkolivegreen', 'lightgreen', 'darkseagreen',
                  'lightcyan', 'saddlebrown', 'deepskyblue', 'slateblue', 'whitesmoke', 'pink', 'darkmagenta',
                  'darkkhaki', 'mediumblue', 'beige', 'blanchedalmond', 'lightsalmon', 'lemonchiffon', 'navajowhite',
                  'darkslateblue', 'lightcoral', 'rosybrown', 'fuchsia', 'peachpuff']

    def __init__(self, log_path):
        self.log_path = log_path + '/loss/'
        self.color_list = Plotter.color_list

        self.markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    @staticmethod
    def compute_best_eps_reward(test_file, eps_size=100):
        average_reward = 0.0
        reward_list = []
        real_sample_used_list = []
        cyber_sample_used_list = []
        count = 0
        with open(file=test_file, mode='r') as f:
            train_data = json.load(fp=f)
            for sample in train_data:
                # if count % 2 == 0:
                #     print(sample['REWARD_SUM'])
                #     pass
                # else:
                #     reward_list.append(sample['REWARD_SUM'])
                #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
                #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

                reward_list.append(sample['REWARD_SUM'])
                real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
                cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])
                #
                # if sample['REWARD_SUM'] >= 100:
                #     print(sample['REWARD_SUM'])
                #     pass
                # else:
                #     reward_list.append(sample['REWARD_SUM'])
                #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
                #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

                count += 1

        min_reward = -100000.0
        pos = 0
        average_reward_list = []
        for i in range(len(reward_list) - eps_size + 1):
            average_reward = sum(reward_list[i: i + eps_size]) / eps_size
            # if average_reward >= 73.0:
            #     print(i, real_sample_used_list[i], cyber_sample_used_list[i], average_reward)
            average_reward_list.append(average_reward)
            # if real_sample_used_list[i] > max_real_sample * 0.5:
            #     if average_reward > min_reward:
            #         min_reward = average_reward
            #         pos = i
            if average_reward > min_reward:
                min_reward = average_reward
                pos = i

        # reward_list.sort(reverse=True)
        # average_reward = sum(reward_list[0:100]) / 100.0
        # print(average_reward)

        return reward_list, real_sample_used_list, cyber_sample_used_list, pos, min_reward, average_reward_list

    def plot_dynamics_env(self):
        test_loss = []
        train_loss = []
        plt.figure(1)
        plt.title('Dynamics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        with open(file=self.log_path + '/DynamicsEnvMlpModel_train_.log', mode='r') as f:
            train_data = json.load(fp=f)
        with open(file=self.log_path + '/DynamicsEnvMlpModel_test_.log', mode='r') as f:
            test_data = json.load(fp=f)
        for sample in train_data:
            train_loss.append(M.log10(sample['DynamicsEnvMlpModel_LOSS']))
        for sample in test_data:
            test_loss.append(M.log10(sample['DynamicsEnvMlpModel_LOSS']))
        times = len(train_loss) // len(test_loss)

        plt.plot([i * times for i in range(len(test_loss))],
                 test_loss,
                 c='g',
                 label='Test Loss')
        plt.plot([i for i in range(len(train_loss))],
                 train_loss,
                 c='b',
                 label='Train loss')
        plt.legend()
        plt.savefig(self.log_path + '/1.png')

    def plot_target_agent(self):
        test_reward = []
        train_cyber_reward = []
        train_real_count = []
        train_real_reward = []
        plt.figure(2)
        plt.title('Target agent reward')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        with open(file=self.log_path + 'TargetAgent_train_.log', mode='r') as f:
            train_data = json.load(fp=f)
            for sample in train_data:
                if sample['ENV'] == 'REAL_ENV':
                    train_real_reward.append(sample['REWARD_MEAN'])
                    train_real_count.append(sample['REAL_SAMPLE_COUNT'])
                else:
                    train_cyber_reward.append(sample['REWARD_MEAN'])
        with open(file=self.log_path + 'TargetAgent_test_.log', mode='r') as f:
            test_data = json.load(fp=f)
            for sample in test_data:
                test_reward.append(sample['REWARD_SUM'])

        times = len(train_cyber_reward) // len(train_real_reward)
        if times == 0:
            times = 1
        # plt.plot([i * times for i in range(len(train_real_reward))], train_real_reward, c='g',
        #          label='Train real reward')
        # plt.plot([i for i in range(len(train_cyber_reward))], train_cyber_reward, c='r', label='Train cyber reward')
        plt.plot([i * times for i in range(len(test_reward))], test_reward, c='b', label='test reward')
        plt.legend()
        # plt.show()
        plt.savefig(self.log_path + '/2.png')

    def plot_ddpg_model(self):
        pass
        actor_loss = []
        critic_loss = []
        with open(file=self.log_path + 'DDPGModel_train_.log', mode='r') as f:
            loss = json.load(fp=f)
            for sample in loss:
                actor_loss.append(sample['DDPGModel_ACTOR'])
                # if sample['DDPGModel_ACTOR'] > 0:
                #     actor_loss.append(M.log10(sample['DDPGModel_ACTOR']))
                # else:
                # actor_loss.append(sample['DDPGModel_ACTOR'])
                critic_loss.append(M.log10(sample['DDPGModel_CRITIC']))
        plt.figure(3)

        plt.subplot(2, 1, 1)

        plt.title('Actor Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.plot([i for i in range(len(actor_loss))], actor_loss, c='g', label='Actor loss')

        plt.subplot(2, 1, 2)

        plt.title('Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot([i for i in range(len(critic_loss))], critic_loss, c='r', label='Critic Loss')

        plt.legend()
        plt.savefig(self.log_path + '/3.png')

    @staticmethod
    def plot_multiply_target_agent_reward_no_show(path_list, save_flag=True, title=None, fig_id=4, label=' ',
                                                  save_path=None):
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen']

        plt.figure(fig_id)
        if title:
            plt.title('Target agent reward ' + title)
        plt.xlabel('Physic system sample')
        plt.ylabel('Reward Sum')
        for i in range(len(path_list)):
            test_reward = []
            real_env_sample_count_index = []
            with open(file=path_list[i] + '/loss/TargetAgent_test_.log', mode='r') as f:
                test_data = json.load(fp=f)
                for sample in test_data:
                    test_reward.append(sample['REWARD_SUM'])
                    real_env_sample_count_index.append(sample['REAL_SAMPLE_COUNT'])

            x_keys = []
            y_values = []
            last_key = real_env_sample_count_index[0]
            last_set = []

            for j in range(len(real_env_sample_count_index)):
                if real_env_sample_count_index[j] == last_key:
                    last_set.append(test_reward[j])
                else:
                    x_keys.append(last_key)
                    y_values.append(last_set)
                    last_key = real_env_sample_count_index[j]
                    last_set = [test_reward[j]]
            y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]
            plt.plot(x_keys, y_values_mean, c=color_list[i], label='Test reward ' + label + str(i))

        plt.legend()
        if save_flag is True:
            for path in path_list:
                plt.savefig(path + '/loss/' + '/compare.png')
        if save_path is not None:
            plt.savefig(save_path)

    @staticmethod
    def plot_multiply_target_agent_reward(path_list, save_flag=True, title=None):
        Plotter.plot_multiply_target_agent_reward_no_show(path_list, save_flag, title)
        plt.show()

    @staticmethod
    def compute_mean_multi_reward(file_list, assemble_flag=False):

        baseline_assemble_reward_list = []
        merged_index_list = []
        for file in file_list:
            merged_reward_list = []
            merged_index_list = []
            file_name = None
            if 'assemble' in file:
                for file_i in glob.glob(file + '/loss/*BEST_AGENT_TEST_REWARD.json'):
                    file_name = file_i
                assert file_name is not None
            else:
                file_name = file + '/loss/TargetAgent_test_.log'

            reward_list, real_sample_used_list, _, pos, min_reward, average_reward_list = \
                Plotter.compute_best_eps_reward(test_file=file_name)
            sum_re = 0
            pre_index = real_sample_used_list[0]
            sum_count = 0
            for re, index in zip(reward_list, real_sample_used_list):
                if index == pre_index:
                    sum_re += re
                    sum_count += 1
                else:
                    print(index)
                    merged_reward_list.append(sum_re / sum_count)
                    merged_index_list.append(pre_index)
                    pre_index = index
                    sum_re = re
                    sum_count = 1
            baseline_assemble_reward_list.append(merged_reward_list)
        baseline_assemble_reward_list = np.mean(np.array(baseline_assemble_reward_list), axis=0)
        baseline_assemble_reward_list_std = np.std(np.array(baseline_assemble_reward_list), axis=0)
        return baseline_assemble_reward_list, merged_index_list, baseline_assemble_reward_list_std

    @staticmethod
    def plot_mean_multiply_target_agent_reward(baseline_list, intel_list, save_path=None):
        baseline_reward_list, baseline_index, baseline_std = Plotter.compute_mean_multi_reward(file_list=baseline_list)
        intel_reward_list, intel_index, intel_std = Plotter.compute_mean_multi_reward(file_list=intel_list)
        a = Plotter(log_path='')
        a.plot_fig(fig_num=4, col_id=1, x=baseline_index, y=baseline_reward_list,
                   title='',
                   x_lable='Number of real data samples used', y_label='Reward', label='Baseline Trainer',
                   marker=Plotter.markers[4])
        plt.fill_between(x=baseline_index,
                         y1=baseline_reward_list - baseline_std,
                         y2=baseline_reward_list + baseline_std,
                         alpha=0.3,
                         facecolor=a.color_list[1],
                         linewidth=0)

        a.plot_fig(fig_num=4, col_id=2, x=intel_index, y=intel_reward_list,
                   title='',
                   x_lable='Number of real data samples used', y_label='Reward', label='Intelligent Trainer',
                   marker=Plotter.markers[8])
        plt.fill_between(x=intel_index,
                         y1=intel_reward_list - intel_std,
                         y2=intel_reward_list + intel_std,
                         alpha=0.3,
                         facecolor=a.color_list[2],
                         linewidth=0)
        if save_path is not None:
            plt.savefig(save_path + '/compare.png', bbox_inces='tight')
            plt.savefig(save_path + '/compare.eps', format='eps', bbox_inces='tight')
            plt.savefig(save_path + '/compare.pdf', format='pdf', bbox_inces='tight')

        plt.show()
        pass

    @staticmethod
    def plot_many_target_agent_reward(path_list, name_list, title=' ', assemble_index=None):
        for i in range(len(path_list)):
            if i == assemble_index:
                assemble_flag = True
            else:
                assemble_flag = False
            baseline_reward_list, baseline_index, baseline_std = Plotter.compute_mean_multi_reward(
                file_list=path_list[i],
                assemble_flag=assemble_flag)
            a = Plotter(log_path='')
            a.plot_fig(fig_num=4, col_id=i, x=baseline_index, y=baseline_reward_list,
                       title=title,
                       x_lable='Number of data samples used', y_label='Reward', label=name_list[i],
                       marker=Plotter.markers[i])
            # plt.fill_between(x=baseline_index,
            #                  y1=baseline_reward_list - baseline_std,
            #                  y2=baseline_reward_list + baseline_std,
            #                  alpha=0.3,
            #                  facecolor=a.color_list[i],
            #                  linewidth=0)
        plt.savefig(path_list[i][0] + '/case1.pdf')
        plt.show()

    @staticmethod
    def plot_multiply_target_agent_reward_MEAN(path_list_list, title="IntelTrainer", max_count=10000, legends=[],
                                               assemble_flag=False):
        color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'cyan', 'plum', 'darkgreen']
        plt.figure(4)
        plt.title('Target agent reward')
        plt.xlabel('Physic system sample')
        plt.ylabel('Reward')
        x_new = np.arange(0, 10000)  #####special set for pen, need to be changed for other cases

        for kkk in range(len(path_list_list)):
            y_new_set = []
            path_list = path_list_list[kkk]
            for i in range(len(path_list)):
                test_reward = []
                real_env_sample_count_index = []
                file_name = None
                if assemble_flag is True:
                    for file in glob.glob(path_list[i] + '/loss/*BEST_AGENT_TEST_REWARD.json'):
                        file_name = file
                    assert file_name is not None
                else:
                    file_name = '/loss/TargetAgent_test_.log'

                with open(file=path_list[i] + file_name, mode='r') as f:
                    test_data = json.load(fp=f)
                    for sample in test_data:
                        test_reward.append(sample['REWARD_MEAN'])
                        real_env_sample_count_index.append(sample['REAL_SAMPLE_COUNT'])

                x_keys = []
                y_values = []
                last_key = real_env_sample_count_index[0]
                last_set = []

                for j in range(len(real_env_sample_count_index)):
                    if real_env_sample_count_index[j] == last_key:
                        last_set.append(test_reward[j])
                    else:
                        x_keys.append(last_key)
                        y_values.append(last_set)
                        last_key = real_env_sample_count_index[j]
                        last_set = [test_reward[j]]
                y_values_mean = [np.mean(y_values[j]) for j in range(len(y_values))]

                f_inter__ = interp1d(x_keys, y_values_mean, fill_value="extrapolate")

                y_new = f_inter__(x_new)
                y_new_set.append(y_new)
            y_new_set = np.asarray(y_new_set)
            y_mean = np.mean(y_new_set, 0)
            y_std = np.std(y_new_set, 0)
            print("y_std=", y_std)
            plt.plot(x_new, y_mean, color_list[kkk])
            plt.fill_between(x_new, y_mean - y_std, y_mean + y_std,
                             alpha=0.5, facecolor=color_list[kkk],
                             linewidth=0)
            # plt.errorbar(x_new, y_mean, yerr=y_std)
            # plt.plot(x_new, y_mean)
        plt.legend(legends)
        plt.title(title)
        for path in path_list:
            plt.savefig(path + '/loss/' + '/' + title + '.png')
        plt.show()

    def plot_error(self, p1, p2):
        test1 = []
        test2 = []
        with open(file=p1, mode='r') as f:
            test_data = json.load(fp=f)
            for data in test_data:
                test1.append(data)

        with open(file=p2, mode='r') as f:
            test_data = json.load(fp=f)
            for data in test_data:
                test2.append(data)
        plt.plot(test1, c=self.color_list[1], label='self')
        plt.plot(test2, c=self.color_list[2], label='benchmark')
        plt.legend()

    def _plot(self, x, y):
        pass

    def plot_fig(self, fig_num, col_id, x, y, title, x_lable, y_label, label=' ', marker='*', label_list=None):
        from pylab import rcParams
        rcParams['figure.figsize'] = 4, 3
        sns.set_style("darkgrid")
        plt.figure(fig_num)
        plt.title(title)
        plt.xlabel(x_lable)
        plt.ylabel(y_label)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

        marker_every = max(int(len(x) / 10), 1)
        # marker_every = 1
        if len(np.array(y).shape) > 1:

            new_shape = np.array(y).shape

            res = np.reshape(np.reshape(np.array([y]), newshape=[-1]), newshape=[new_shape[1], new_shape[0]],
                             order='F').tolist()
            res = list(res)
            for i in range(len(res)):
                res_i = res[i]
                if label_list:
                    label_i = label_list[i]
                else:
                    label_i = label + '_' + str(i)
                plt.subplot(len(res), 1, i + 1)
                plt.title(title + '_' + str(i))
                plt.plot(x, res_i, self.color_list[col_id], label=label_i, marker=marker,
                         markevery=marker_every, markersize=6, linewidth=1)
                # sns.lmplot(x, res_i, self.color_list[col_id], label=label + '_' + str(i), marker=marker, markevery=marker_every, markersize=24)
                col_id += 1
        else:
            plt.plot(x, y, self.color_list[col_id], label=label, marker=marker, markevery=marker_every, markersize=6,
                     linewidth=1)
        plt.legend()

    def log_intel_action(self, path):
        action = []
        obs = []
        val_loss = []
        val_loss_change = []
        contrl_loss = []
        control_loss_change = []
        reward = []
        with open(path, 'r') as f:
            content = json.load(fp=f)
            for sample in content:
                action.append(sample['ACTION'])
                obs.append(sample['OBS'])
                val_loss.append(sample['VALUE_FUNCTION_LOSS'])
                val_loss_change.append(sample['VALUE_FUNCTION_LOSS_CHANGE'])
                contrl_loss.append(sample['CONTROLLER_LOSS_CHANGE'])
                control_loss_change.append(sample['CONTROLLER_LOSS_CHANGE'])
                reward.append(sample['REWARD'])
        return action, obs, val_loss, val_loss_change, contrl_loss, control_loss_change, reward

    def plot_intel_action(self, path):
        action, obs, val_loss, val_loss_change, contrl_loss, control_loss_change, reward = self.log_intel_action(
            path=path)
        self.plot_fig(fig_num=1, col_id=1, x=[i for i in range(len(action))],
                      y=action, title='action', x_lable='', y_label='')

        self.plot_fig(fig_num=2, col_id=2, x=[i for i in range(len(obs))],
                      y=obs, title='obs', x_lable='', y_label='')

        self.plot_fig(fig_num=3, col_id=3, x=[i for i in range(len(val_loss))],
                      y=val_loss, title='val_loss', x_lable='', y_label='')

        self.plot_fig(fig_num=4, col_id=4, x=[i for i in range(len(val_loss_change))],
                      y=val_loss_change, title='val_loss_change', x_lable='', y_label='')

        self.plot_fig(fig_num=5, col_id=5, x=[i for i in range(len(contrl_loss))],
                      y=contrl_loss, title='contrl_loss', x_lable='', y_label='')

        self.plot_fig(fig_num=6, col_id=6, x=[i for i in range(len(control_loss_change))],
                      y=control_loss_change, title='control_loss_change', x_lable='', y_label='')

        self.plot_fig(fig_num=7, col_id=7, x=[i for i in range(len(reward))],
                      y=reward, title='reward', x_lable='', y_label='')

    def plot_intel_actions(self, path_list):
        action_list = []
        for iii in range(len(path_list)):
            path = path_list[iii] + '/loss/TrainerEnv_train_.log'
            action = []
            obs = []
            val_loss = []
            val_loss_change = []
            contrl_loss = []
            control_loss_change = []
            reward = []
            with open(path, 'r') as f:
                content = json.load(fp=f)
                for sample in content:
                    action.append(sample['ACTION'])
                    obs.append(sample['OBS'])
                    val_loss.append(sample['VALUE_FUNCTION_LOSS'])
                    val_loss_change.append(sample['VALUE_FUNCTION_LOSS_CHANGE'])
                    contrl_loss.append(sample['CONTROLLER_LOSS_CHANGE'])
                    control_loss_change.append(sample['CONTROLLER_LOSS_CHANGE'])
                    reward.append(sample['REWARD'])
            action = np.asarray(action)
            action_list.append(action)
        mean_action = 1.0 * action_list[0]
        for iii in range(len(action_list) - 1):
            mean_action += action_list[iii + 1]
        mean_action /= len(action_list)
        action_list = np.asarray(action_list)
        x = [i for i in range(len(mean_action))]
        std_action = np.std(action_list, 0)
        for col in range(3):
            plt.figure()
            plt.plot(x, mean_action[:, col])
            plt.fill_between(x, mean_action[:, col] - std_action[:, col], mean_action[:, col] + std_action[:, col])
        plt.show()


if __name__ == '__main__':
    from log.baselineTestLog import LOG

    #####pen results
    # path_list_pen = ["/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-36-21",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-45-07",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-51-28",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-57-36",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_03-03-46",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-36-46",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-45-42",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-52-06",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_02-58-28",
    #                     "/home/liyuanl/MRL_new/log/Pendulum-v0/2018-04-20_03-04-49"]
    #
    # #####plot the old mountain car results
    # path_list = ["/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_02-44-41",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_04-11-39",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_05-47-30",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_02-45-31",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_04-24-29",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_06-16-05",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_02-46-04",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_04-07-21",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_05-39-30",
    #             "/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_07-11-11",]
    #
    # #path_list_2 = ["/home/liyuanl/MRL_new/log/MountainCarContinuous-v0/2018-04-20_10-42-00"]
    # Plotter.plot_multiply_target_agent_reward(path_list_pen)
    Moun_base_path_list = ["/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-23_22-45-10",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-23_23-04-34",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-23_23-24-11",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-23_23-43-39",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_00-02-59",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_00-22-16",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_00-41-35",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_01-00-51",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_01-20-11",
                           "/home/liyuanl/MRL_new/log/baselineTestLog/MountainCarContinuous-v0/2018-04-24_01-39-31", ]
    Plotter.plot_multiply_target_agent_reward(Moun_base_path_list)
    plt.show()
