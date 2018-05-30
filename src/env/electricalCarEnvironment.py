from src.env.gridMapEnvironment import GridMapEnvironment
from src.config.config import Config
from config.key import CONFIG_KEY
from copy import deepcopy


class ElectricalCarEnvironment(GridMapEnvironment):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/electricalCarEnvironmentKey.json')

    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def get_state(self, env):
        state = self._electrical_car_convert_state_to_list(state=self.state)
        return state

    def step(self, action):
        if self.reset_flag is True:
            self.reset()
        temp_state = deepcopy(self.state)
        state_list, reward, reset_flag = super().step(action)

        state = self._convert_list_to_state(state_list=state_list)

        state['LEFT_POWER'] = temp_state['LEFT_POWER']
        state['X_CHARGE_SITE_1'] = temp_state['X_CHARGE_SITE_1']
        state['Y_CHARGE_SITE_1'] = temp_state['Y_CHARGE_SITE_1']
        state['X_CHARGE_SITE_2'] = temp_state['X_CHARGE_SITE_2']
        state['Y_CHARGE_SITE_2'] = temp_state['Y_CHARGE_SITE_2']

        state['LEFT_POWER'] -= 1

        if self._reach_charge_site(state=state):
            print("reach charge site!")
            state['LEFT_POWER'] = self.config.config_dict['FULL_POWER']
            reward += self.config.config_dict['REACH_CHARGE_SITE_REWARD']

        if state['LEFT_POWER'] <= 0:
            print("out of power")
            reward -= self.config.config_dict['OUT_OF_POWER_COST']
            reset_flag = True
        self.state = state
        self.reset_flag = reset_flag
        return self._electrical_car_convert_state_to_list(self.state), reward, self.reset_flag

    def reset(self):
        self.state['T'] = 0
        self.state['X'] = self.config.config_dict['START'][0]
        self.state['Y'] = self.config.config_dict['START'][1]
        self.state['DIRECTION'] = self.config.config_dict['INIT_DIRECTION']
        self.state['LEFT_POWER'] = self.config.config_dict['FULL_POWER']

        self.state['X_CHARGE_SITE_1'] = self.config.config_dict['CHARGE_SITE_1'][0]
        self.state['Y_CHARGE_SITE_1'] = self.config.config_dict['CHARGE_SITE_1'][1]

        self.state['X_CHARGE_SITE_2'] = self.config.config_dict['CHARGE_SITE_2'][0]
        self.state['Y_CHARGE_SITE_2'] = self.config.config_dict['CHARGE_SITE_2'][1]

        self.reset_flag = False
        return self.get_state(env=self)

    def computer_reward(self, bound_hit_flag, old_state, new_state):
        return super().computer_reward(bound_hit_flag, old_state, new_state)

    def _electrical_car_convert_state_to_list(self, state):
        super()._convert_state_to_list(state=state)
        s = [state['X'],
             state['Y'],
             state['T'],
             state['DIRECTION'],
             state['LEFT_POWER'],
             state['X_CHARGE_SITE_1'],
             state['Y_CHARGE_SITE_1'],
             state['X_CHARGE_SITE_2'],
             state['Y_CHARGE_SITE_2']]
        return s

    @staticmethod
    def _convert_list_to_state(state_list):
        key = ["X", "Y", "T", "DIRECTION",
               "LEFT_POWER", "X_CHARGE_SITE_1", "Y_CHARGE_SITE_1",
               "X_CHARGE_SITE_2", "Y_CHARGE_SITE_2", "LEFT_POWER"]

        state = {}
        for key_i, val in zip(key, state_list):
            state[key_i] = val
        return state

    @staticmethod
    def _reach_charge_site(state):
        if state['X'] == state['X_CHARGE_SITE_1'] and state['Y'] == state['Y_CHARGE_SITE_1']:
            return True
        if state['X'] == state['X_CHARGE_SITE_2'] and state['Y'] == state['Y_CHARGE_SITE_2']:
            return True
        return False
