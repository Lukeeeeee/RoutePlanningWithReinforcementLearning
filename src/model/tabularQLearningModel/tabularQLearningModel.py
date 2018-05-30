from src.model.model import Model
import numpy as np
from config.key import CONFIG_KEY
from src.config.config import Config
from src.util.util import SamplerData


class TabularQLearningModel(Model):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/tabularQLearingModelKey.json')

    def __init__(self, config):
        super().__init__(config)
        self.q_table = np.zeros(shape=self.config.config_dict[['Q_TABULAR_SHAPE']])
        self.sample_data = SamplerData()

    def return_table_value(self, action, state):
        return self.q_table[action][state[0]][state[1]]

    def set_table_value(self, action, state, val):
        self.q_table[action][state[0]][state[1]] = val

    def update(self):
        sample_data = self.sample_data
        for sample in sample_data:
            if sample['DONE'] is True:
                target_q = sample['REWARD']
            else:
                max_q_action = 0
                for i in range(self.config.config_dict['ACTION_SIZE']):
                    if self.return_table_value(action=i, state=sample['NEW_STATE']) > \
                            self.return_table_value(action=max_q_action, state=sample['NEW_STATE']):
                        max_q_action = i

                target_q = sample['REWARD'] + self.config.config_dict['GAMMA'] * self.return_table_value(
                    action=max_q_action, state=sample['NEW_STATE'])

            update_val = (1 - self.config.config_dict['LEARNING_RATE']) * self.return_table_value(
                action=sample['ACTION'], state=sample['STATE']) + self.config.config_dict['LEARNING_RATE'] * target_q

            self.set_table_value(action=sample['ACTION'], state=sample['STATE'], val=update_val)
        self.sample_data.reset()

    def predict(self, state):
        max_q_action = 0
        for i in range(self.config.config_dict['ACTION_SIZE']):
            if self.return_table_value(action=i, state=state) > self.return_table_value(action=max_q_action,
                                                                                        state=state):
                max_q_action = i
        return max_q_action

    def store_one_sample(self, state, next_state, action, reward, done):
        # state = state[0:2]
        # new_state = next_state[0:2]
        self.sample_data.append(state=state,
                                action=action,
                                reward=reward,
                                done=done,
                                new_state=next_state)
