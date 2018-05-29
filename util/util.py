from src.agent.targetAgent.targetAgent import TargetAgent
from src.env.gridMapEnvironment import GridMapEnvironment
from src.model.tabularQLearningModel.tabularQLearningModel import TabularQLearningModel
from src.config.config import Config
from copy import deepcopy
from src.util.sampler.sampler import Sampler
from src.core import GamePlayer


def load_config(key_list, config_path):
    conf = Config(standard_key_list=key_list)
    conf.load_config(path=config_path)
    return conf


def create_target_agent(config_path, env, model):
    target_agent_config = load_config(key_list=TargetAgent.key_list,
                                      config_path=config_path)

    a = TargetAgent(config=target_agent_config,
                    real_env=env,
                    cyber_env=None,
                    model=model,
                    sampler=Sampler())
    return a


def create_simple_grid_environment(config_path):
    env_config = load_config(key_list=GridMapEnvironment.key_list,
                             config_path=config_path)
    env = GridMapEnvironment(config=env_config)

    return env


def create_tabular_q_learning_model(config_path):
    model_config = load_config(key_list=TabularQLearningModel.key_list, config_path=config_path)
    model = TabularQLearningModel(config=model_config)
    return model


def create_game_player(config_path, env, agent, basic_list):
    player_config = load_config(key_list=GamePlayer.key_list,
                                config_path=config_path)

    player = GamePlayer(config=player_config, env=env, agent=agent, basic_list=basic_list)
    return player