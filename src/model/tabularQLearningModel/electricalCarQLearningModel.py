from src.model.tabularQLearningModel.tabularQLearningModel import TabularQLearningModel
from src.config.config import Config
from config.key import CONFIG_KEY


class ElectricalCarQLearningModel(TabularQLearningModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/tabularQLearingModelKey.json')

    def return_table_value(self, action, state):
        return self.q_table[action][state[0]][state[1]][state[4]][state[5]][state[6]][state[7]][state[8]]

    def set_table_value(self, action, state, val):
        self.q_table[action][state[0]][state[1]][state[4]][state[5]][state[6]][state[7]][state[8]] = val
