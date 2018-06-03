import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

from config.config_SimpleGridWorld import CONFIG_SET_SIMPLE_GRID_WORLD
from util import util as test_util


def create_case_1_game():
    n = 4
    m = 4
    env = test_util.create_simple_grid_environment(config_path=CONFIG_SET_SIMPLE_GRID_WORLD + '/environmentConfig.json')

    tabular_model = test_util.create_tabular_q_learning_model(
        config_path=CONFIG_SET_SIMPLE_GRID_WORLD + '/modelConfig.json',
        n=env.config.config_dict['N'],
        m=env.config.config_dict['M'])
    agent = test_util.create_target_agent(config_path=CONFIG_SET_SIMPLE_GRID_WORLD + '/agentConfig.json',
                                          env=env,
                                          model=tabular_model)
    player = test_util.create_game_player(config_path=CONFIG_SET_SIMPLE_GRID_WORLD + '/gamePlayerTestConfig.json',
                                          env=env,
                                          agent=agent,
                                          basic_list=[tabular_model, env, agent])
    return player


if __name__ == '__main__':
    player = create_case_1_game()
    player.play()
    player.print_log_to_file()
