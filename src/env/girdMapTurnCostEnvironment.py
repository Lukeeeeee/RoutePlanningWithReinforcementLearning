from src.env.gridMapEnvironment import GridMapEnvironment
from src.config.config import Config
from config.key import CONFIG_KEY


class GridMapTurnCostEnvironment(GridMapEnvironment):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/turnCostGridWorldEnvironmentKey.json')

    def __init__(self, config):
        super().__init__(config)
        self.turn_cost_dict = {
            "S": ("E", "N"),
            "N": ("W", "S"),
            "W": ("S", "E"),
            "E": ("N", "W")
        }

    def computer_reward(self, bound_hit_flag, old_state, new_state):
        reward = super().computer_reward(bound_hit_flag, old_state, new_state)
        if new_state['DIRECTION'] in self.turn_cost_dict[old_state['DIRECTION']]:
            reward -= self.config.config_dict['TURN_LEFT_COST']
        return reward
