from src.core import Basic
import numpy as np
import easy_tf_log
from src.config.config import Config


class SamplerData(object):
    def __init__(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def reset(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state, action, new_state, done, reward):
        self.state_set.append(state)
        self.new_state_set.append(new_state)
        self.reward_set.append(reward)
        self.done_set.append(done)
        self.action_set.append(action)
        self.cumulative_reward += reward

    def union(self, sample_data):
        self.state_set += sample_data.state_set
        self.new_state_set += sample_data.new_state_set
        self.reward_set += sample_data.reward_set
        self.done_set += sample_data.done_set
        self.action_set += sample_data.action_set
        self.cumulative_reward += sample_data.cumulative_reward
        self.step_count_per_episode += sample_data.step_count_per_episode


class Sampler(Basic):
    def __init__(self, cost_fn=None, config=None):
        super().__init__(config=config)
        if self.config is None:
            self.config = Config(standard_key_list=['REAL_ENVIRONMENT_STATUS',
                                                    'CYBER_ENVIRONMENT_STATUS', 'TEST_ENVIRONMENT_STATUS'])
        self.config.config_dict['REAL_ENVIRONMENT_STATUS'] = 1
        self.config.config_dict['CYBER_ENVIRONMENT_STATUS'] = 0
        self.config.config_dict['TEST_ENVIRONMENT_STATUS'] = 2

        self._test_data = SamplerData()
        self._cyber_data = SamplerData()
        self._real_data = SamplerData()
        self.cost_fn = cost_fn

        self.data = None

        self._env_status = None

        self.env_status = 1

    def sample(self, env, agent, sample_count, store_flag=False, agent_print_log_flag=False, reset_Flag=True):
        if agent.status == agent.status_key['TEST']:
            state = env.reset()
        else:
            state = env.get_state(env)
        sample_record = SamplerData()
        for i in range(sample_count):

            if agent.status == agent.status_key['TEST']:
                print(state)
            action = agent.predict(state=state)
            new_state, re, done = env.step(action)
            if not isinstance(done, bool):
                if done[0] == 1:
                    done = True
                else:
                    done = False
            self.step_count_per_episode += 1
            agent.env_sample_count += 1
            if store_flag is True:
                agent.store_one_sample(state=state,
                                       action=action,
                                       next_state=new_state,
                                       reward=re,
                                       done=done)

            self.data.append(state=state,
                             action=action,
                             new_state=new_state,
                             done=done,
                             reward=re)

            sample_record.append(state=state,
                                 action=action,
                                 reward=re,
                                 new_state=new_state,
                                 done=done)

            self.log_every_step(agent=agent,
                                reward=re)

            if done is True:
                tmp_state_set = self.state_set + [self.new_state_set[-1]]
                self.log_every_episode(agent=agent,
                                       average_reward=self.cumulative_reward / self.step_count_per_episode,
                                       reward=self.cumulative_reward,
                                       state_set=tmp_state_set,
                                       action_set=self.action_set,
                                       agent_print_log_flag=agent_print_log_flag)
                state = self.reset(env, agent, reset_Noise=reset_Flag)
                agent.log_queue.queue.clear()
            else:
                state = new_state

        return sample_record

    def reset(self, env, agent, reset_Noise=True):
        if reset_Noise == True:
            agent.reset()
        # TODO WHEN DONE IS GOT, THE ENV WILL RESET  BY IT SELF?
        return env.reset()

    def train(self, *args, **kwargs):
        pass

    def print_log_queue(self, status):
        pass

    def log_every_step(self, agent, reward, *arg, **kwargs):

        agent.log_queue.put({agent.name + '_SAMPLE_REWARD': reward})

    def log_every_episode(self, agent, average_reward, reward, state_set, action_set, agent_print_log_flag=False):
        if agent_print_log_flag is True:
            agent.print_log_queue(status=agent.status)

    def set_F(self, F1, F2):
        self.F1 = F1
        self.F2 = F2

    @property
    def env_status(self):
        return self._env_status

    @env_status.setter
    def env_status(self, new):
        self._env_status = new
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.data = self._real_data
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self.data = self._cyber_data
        elif self._env_status == self.config.config_dict['TEST_ENVIRONMENT_STATUS']:
            self.data = self._test_data

    @property
    def current_env_status(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return 'REAL_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return 'CYBER_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['TEST_ENVIRONMENT_STATUS']:
            return 'TEST_ENVIRONMENT_STATUS'

    @property
    def state_set(self):
        return self.data.state_set

    @state_set.setter
    def state_set(self, new_val):
        self.data.state_set = new_val

    @property
    def action_set(self):
        return self.data.action_set

    @action_set.setter
    def action_set(self, new_val):
        self.data.action_set = new_val

    @property
    def reward_set(self):
        return self.data.reward_set

    @reward_set.setter
    def reward_set(self, new_val):
        self.data.reward_set = new_val

    @property
    def new_state_set(self):
        return self.data.new_state_set

    @new_state_set.setter
    def new_state_set(self, new_val):
        self.data.new_state_set = new_val

    @property
    def cumulative_reward(self):
        return self.data.cumulative_reward

    @cumulative_reward.setter
    def cumulative_reward(self, new_val):
        self.data.cumulative_reward = new_val

    @property
    def step_count_per_episode(self):
        return self.data.step_count_per_episode

    @step_count_per_episode.setter
    def step_count_per_episode(self, new_val):
        self.data.step_count_per_episode = new_val

    @property
    def done_set(self):
        return self.data.done_set

    @done_set.setter
    def done_set(self, new_val):
        self.data.done_set = new_val
