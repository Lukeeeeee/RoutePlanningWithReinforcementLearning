import numpy as np


def noise_adder(action, agent):
    noise = []
    if agent.config.config_dict['NOISE_FLAG'] == 2 and agent._real_env_sample_count < agent.config.config_dict[
        'MAX_SAMPLE_COUNT'] * 0.5:
        p = (agent._real_env_sample_count // 1000) / (agent.config.config_dict['MAX_SAMPLE_COUNT'] * 0.5 / 1000.0)
        noise = (1 - p) * agent.model.action_noise()
        action = p * action + noise

    elif agent.config.config_dict['NOISE_FLAG'] == 1:
        ep = agent._real_env_sample_count / agent.config.config_dict['MAX_SAMPLE_COUNT'] * \
             agent.config.config_dict['EP_MAX']
        noise_scale = (agent.config.config_dict['INIT_NOISE_SCALE'] * agent.config.config_dict['NOISE_DECAY'] ** ep) * \
                      (agent.real_env.action_space.high - agent.real_env.action_space.low)
        noise = noise_scale * agent.model.action_noise()
        action = action + noise
    elif agent.config.config_dict['NOISE_FLAG'] == 3:
        noise = agent.model.action_noise()
        action = action + noise
    noise = np.reshape(np.array([noise]), [-1]).tolist()

    # action = np.clip(action, [-1], [1])
    # print("action", action)
    action = np.clip(action, [-1], [1])
    return action, noise
