from src.env.env import BasicEnv
from src.config.config import Config
from config.key import CONFIG_KEY
import math
import numpy as np
from copy import deepcopy


class GridMapEnvironment(BasicEnv):
    standard_key_list = Config.load_json(CONFIG_KEY + '/simpleGridWorldEnvironmentKey.json')

    def __init__(self, config):
        super(GridMapEnvironment, self).__init__(config=config)
        self.move_vector = {'S': {'X': -1, 'Y': 0},
                            'N': {'X': 1, 'Y': 0},
                            'W': {'X': 0, 'Y': 1},
                            'E': {'X': 0, 'Y': -1}}
        self.reset_flag = False
        self.vec_set = []
        self.edge_set = []
        self.state = {}
        self.reset()

    def step(self, action):
        if self.reset_flag is True:
            self.reset()
        new_x, new_y, bound_hit_flag, new_direction = self.computer_new_loc(direction=self.state['DIRECTION'],
                                                                            action=action,
                                                                            x=self.state['X'],
                                                                            y=self.state['Y'])
        print("x, y", new_x, new_y)

        if new_x == self.config.config_dict['TARGET'][0] and new_y == self.config.config_dict['TARGET'][1]:
            reach_target_flag = True
            print("Reach target point!!!")
        else:
            reach_target_flag = False
        self.reset_flag = reach_target_flag or self.state['T'] + 1 > self.config.config_dict['MAX_STEP']

        reward = self.computer_reward(bound_hit_flag=bound_hit_flag, x=self.state['X'], y=self.state['Y'], new_x=new_x,
                                      new_y=new_y)
        new_state = {
            'T': self.state['T'] + 1,
            'X': new_x,
            'Y': new_y,
            'DIRECTION': new_direction
        }

        self.state = new_state

        return self._convert_state_to_list(new_state), reward, self.reset_flag

    def computer_new_loc(self, direction, action, x, y):

        new_direction = action
        new_x = x + self.move_vector[new_direction]['X']
        new_y = y + self.move_vector[new_direction]['Y']
        if new_x < 0 or new_x >= self.config.config_dict['N'] or new_y < 0 or new_y >= self.config.config_dict['M']:
            return x, y, True, new_direction
        else:
            return new_x, new_y, False, new_direction

    # def compute_new_direction(self, direction, action):
    #
    #     # ACTION = {N, W, S ,E}
    #     # DIRECTION = {N, W, S, E}
    #
    #     if action is 'S':
    #         return direction
    #     if action is 'L':
    #         if direction is 'N':
    #             return 'W'
    #         if direction is 'W':
    #             return 'S'
    #         if direction is 'S':
    #             return 'E'
    #         if direction is 'E':
    #             return 'N'
    #     if action is 'R':
    #         if direction is 'N':
    #             return 'E'
    #         if direction is 'E':
    #             return 'S'
    #         if direction is 'S':
    #             return 'W'
    #         if direction is 'W':
    #             return 'N'
    #     if action is 'T':
    #         if direction is 'N':
    #             return 'S'
    #         if direction is 'E':
    #             return 'W'
    #         if direction is 'S':
    #             return 'N'
    #         if direction is 'W':
    #             return 'E'

    def compute_cost(self, x, y, new_x, new_y, t, action, power):
        raise NotImplementedError

    def computer_reward(self, bound_hit_flag, x, y, new_x, new_y):

        pre_dist = self._two_point_distance(p1=(x, y), p2=self.config.config_dict['TARGET'])
        new_dist = self._two_point_distance(p1=(new_x, new_y), p2=self.config.config_dict['TARGET'])

        reward = np.sign(new_dist - pre_dist) - self.config.config_dict['NORMAL_MOVING_COST'] + int(bound_hit_flag) * (
            -10)
        if new_x == self.config.config_dict['TARGET'][0] and new_y == self.config.config_dict['TARGET'][1]:
            reward += 100
        return reward

    def reset(self):
        self.state['T'] = 0
        self.state['X'] = self.config.config_dict['START'][0]
        self.state['Y'] = self.config.config_dict['START'][1]
        self.state['DIRECTION'] = self.config.config_dict['INIT_DIRECTION']
        self.reset_flag = False
        return self.get_state(env=self)

    @staticmethod
    def _two_point_distance(p1, p2):
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    def get_state(self, env):
        state = self._convert_state_to_list(state=self.state)
        return state

    @staticmethod
    def _convert_state_to_list(state):
        s = [state['X'], state['Y'], state['T'], state['DIRECTION']]
        return s
