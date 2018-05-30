import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from config.config_TurnCostGridWorld import CONFIG_SET_TURN_COST_GRID_WORLD
from util import util as test_util


def create_case_2_game():
    n = 4
    m = 4

    tabular_model = test_util.create_turn_cost_tabular_q_learning_model(
        config_path=CONFIG_SET_TURN_COST_GRID_WORLD + '/modelConfig.json')
    env = test_util.create_turn_cost_simple_grid_environment(
        config_path=CONFIG_SET_TURN_COST_GRID_WORLD + '/environmentConfig.json')
    agent = test_util.create_target_agent(config_path=CONFIG_SET_TURN_COST_GRID_WORLD + '/agentConfig.json',
                                          env=env,
                                          model=tabular_model)
    player = test_util.create_game_player(config_path=CONFIG_SET_TURN_COST_GRID_WORLD + '/gamePlayerTestConfig.json',
                                          env=env,
                                          agent=agent,
                                          basic_list=[tabular_model, env, agent])
    return player


if __name__ == '__main__':
    player = create_case_2_game()
    player.play()
