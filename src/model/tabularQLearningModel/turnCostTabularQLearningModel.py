from src.model.tabularQLearningModel.tabularQLearningModel import TabularQLearningModel
from src.config.config import Config
from config.key import CONFIG_KEY


class TurnCostTabularQLearningModel(TabularQLearningModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/tabularQLearingModelKey.json')

    def __init__(self, config):
        super().__init__(config)
        self.direction_map_to_scalar = {
            'N': 0,
            'W': 1,
            'S': 2,
            'E': 3
        }

    def return_table_value(self, action, state):
        return self.q_table[action][state[0]][state[1]][self.direction_map_to_scalar[state[3]]]

    def set_table_value(self, action, state, val):
        self.q_table[action][state[0]][state[1]][self.direction_map_to_scalar[state[3]]] = val
