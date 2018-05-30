from src.model.tabularQLearningModel.tabularQLearningModel import TabularQLearningModel
from src.config.config import Config
from config.key import CONFIG_KEY


class TurnCostTabularQLearningModel(TabularQLearningModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/turnCostGridWorldEnvironmentKey.json')

    def __init__(self, config):
        super().__init__(config)

    def return_table_value(self, action, state):
        return self.q_table[action][state[0]][state[1]][state[3]]

    def set_table_value(self, action, state, val):
        self.q_table[action][state[0]][state[1]][state[3]] = val
