from src.model.model import Model
import numpy as np
from config.key import CONFIG_KEY
from src.config.config import Config
from src.util.util import SamplerData


class TabularQLearningModel(Model):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/tabularQLearingModelKey.json')

    def __init__(self, config, data=None):
        super().__init__(config, data)
        self.action_size = self.config.config_dict['ACTION_SIZE']
        self.state_size = self.config.config_dict['N'] * self.config.config_dict['M']
        self.q_table_size = self.action_size * self.state_size
        self.q_table = np.zeros(
            shape=[self.config.config_dict['ACTION_SIZE'], self.config.config_dict['N'], self.config.config_dict['M']])
        self.sample_data = SamplerData()

    def update(self):
        sample_data = self.sample_data
        for sample in sample_data:
            if sample['DONE'] is True:
                target_q = sample['REWARD']
            else:
                max_q_action = 0
                for i in range(self.config.config_dict['ACTION_SIZE']):
                    if self.q_table[i][sample['NEW_STATE'][0]][sample['NEW_STATE'][1]] > \
                            self.q_table[max_q_action][sample['NEW_STATE'][0]][sample['NEW_STATE'][1]]:
                        max_q_action = i

                target_q = sample['REWARD'] + self.config.config_dict['GAMMA'] * \
                                              self.q_table[max_q_action][sample['NEW_STATE'][0]][sample['NEW_STATE'][1]]

            update_val = (1 - self.config.config_dict['LEARNING_RATE']) * \
                         self.q_table[sample['ACTION']][sample['STATE'][0]][sample['STATE'][1]] + \
                         self.config.config_dict['LEARNING_RATE'] * target_q

            self.q_table[sample['ACTION']][sample['STATE'][0]][sample['STATE'][1]] = update_val
        self.sample_data.reset()

    def predict(self, state):
        state = state[0:2]
        # index = self._convert_s_a_to_index(state=state, action=0)
        max_q_action = 0
        for i in range(self.config.config_dict['ACTION_SIZE']):
            if self.q_table[i][state[0]][state[1]] > \
                    self.q_table[max_q_action][state[0]][state[1]]:
                max_q_action = i
        return max_q_action

    # def _convert_s_a_to_index(self, state, action):
    #     index = state[0] + state[1] * self.config.config_dict['N']
    #     return index, action
    #
    # def _convert_index_to_s_a(self, index):
    #
    #     action = index[1]
    #     state_1 = index // self.config.config_dict['N']
    #     state_0 = index % self.config.config_dict['N']
    #     return (state_0, state_1), action

    def store_one_sample(self, state, next_state, action, reward, done):
        state = state[0:2]
        new_state = next_state[0:2]
        self.sample_data.append(state=state,
                                action=action,
                                reward=reward,
                                done=done,
                                new_state=new_state)
