from src.agent.agent import Agent
from src.config.config import Config
from config.key import CONFIG_KEY
import numpy as np
from src.util.sampler.sampler import Sampler
import easy_tf_log
from src.core import Basic
from src.util.noiseAdder import noise_adder


class TargetAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/agentKey.json')

    def __init__(self, config, real_env, cyber_env, model, sampler=Sampler()):
        super(TargetAgent, self).__init__(config=config,
                                          env=real_env,
                                          model=model,
                                          sampler=sampler)
        self.real_env = real_env
        self.cyber_env = cyber_env
        self.env = None
        self._env_status = self.config.config_dict['REAL_ENVIRONMENT_STATUS']
        self._real_env_sample_count = 0
        self._cyber_env_sample_count = 0
        self.SamplerTraingCount = 0
        self.action_dict = ['S', 'N', 'W', 'E']

    @property
    def env_sample_count(self):
        if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return self._real_env_sample_count
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return self._cyber_env_sample_count

    @env_sample_count.setter
    def env_sample_count(self, new_value):
        if self.status == self.status_key['TEST']:
            return
        if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self._real_env_sample_count = new_value
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self._cyber_env_sample_count = new_value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_value):
        if new_value != Basic.status_key['TRAIN'] and new_value != Basic.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)
        if new_value == Basic.status_key['TEST'] and self.env_status == self.config.config_dict[
            'REAL_ENVIRONMENT_STATUS']:
            self.sampler.env_status = ['TEST_ENVIRONMENT_STATUS']
        if self._status == new_value:
            return
        self._status = new_value
        self.model.status = new_value

    @property
    def env_status(self):
        return self._env_status

    @env_status.setter
    def env_status(self, new_sta):
        self._env_status = new_sta
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.env = self.real_env
            self.model.env_status = self.model.config.config_dict['REAL_ENVIRONMENT_STATUS']
            if self.status == self.status_key['TEST']:
                self.sampler.env_status = self.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']
            else:
                self.sampler.env_status = self.sampler.config.config_dict['REAL_ENVIRONMENT_STATUS']
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self.env = self.cyber_env
            self.model.env_status = self.model.config.config_dict['CYBER_ENVIRONMENT_STATUS']
            self.sampler.env_status = self.sampler.config.config_dict['CYBER_ENVIRONMENT_STATUS']

        else:
            raise ValueError('Wrong Agent Environment Env Status: %d' % new_sta)

    @property
    def current_env_status(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return 'REAL_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return 'CYBER_ENVIRONMENT_STATUS'

    def predict(self, state, *args, **kwargs):
        count = self._real_env_sample_count
        eps = 1.0 - (self.config.config_dict['EPS'] - self.config.config_dict['EPS_GREEDY_FINAL_VALUE']) * \
                    (count / self.config.config_dict['EPS_ZERO_FLAG'])
        if eps < 0:
            eps = 0.0
        rand_eps = np.random.rand(1)
        if self.config.config_dict['EPS_GREEDY_FLAG'] == 1 and rand_eps < eps and self.status == self.status_key[
            'TRAIN']:
            res = np.random.randint(low=0, high=4)
        else:
            res = np.array(self.model.predict(state))

        res = self.action_dict[res]
        return res

    def sample(self, env, sample_count, store_flag=False, agent_print_log_flag=False, resetNoise_Flag=False):
        return super().sample(env, sample_count, store_flag, agent_print_log_flag, resetNoise_Flag)

    def train(self):
        res_dict = self.model.update()
        return res_dict

    def store_one_sample(self, state, next_state, action, reward, done):
        for i in range(len(self.action_dict)):
            if self.action_dict[i] == action:
                action = i
        self.model.store_one_sample(state=state,
                                    next_state=next_state,
                                    action=action,
                                    reward=reward,
                                    done=done)

    def return_most_recent_sample(self, sample_count, memory):
        pass

    def init(self):
        self.model.init()
        self.model.reset()
        super().init()

    def print_log_queue(self, status):
        self.status = status
        reward_list = []
        while self.log_queue.qsize() > 0:
            reward_list.append(self.log_queue.get()[self.name + '_SAMPLE_REWARD'])
        if len(reward_list) > 0:
            reward_list = np.array(reward_list)
            sum = np.sum(reward_list).item()
            mean = np.mean(reward_list).item()
            std = np.mean(reward_list).item()

            env_status = None
            if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
                env_status = 'REAL_ENV'
            elif self.env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
                env_status = 'CYBER_ENV'
            print("%s %s Reward: Sum: %f Average %f Std %f" % (self.name, env_status, sum, mean, std))

            if self.status == self.status_key['TRAIN']:
                self.log_file_content.append({'INDEX': self.log_print_count,
                                              'REWARD_SUM': sum,
                                              'REWARD_MEAN': mean,
                                              'REWARD_STD': std,
                                              'REAL_SAMPLE_COUNT': self._real_env_sample_count,
                                              'CYBER_SAMPLE_COUNT': self._cyber_env_sample_count,
                                              'ENV': env_status})
            elif self.status == self.status_key['TEST']:
                self.log_file_content.append({'INDEX': self.log_print_count,
                                              'REWARD_SUM': sum,
                                              'REWARD_MEAN': mean,
                                              'REWARD_STD': std,
                                              'REAL_SAMPLE_COUNT': self._real_env_sample_count,
                                              'CYBER_SAMPLE_COUNT': self._cyber_env_sample_count,
                                              'ENV': env_status})
            self.log_print_count += 1
        # TODO HOW TO ELEGANT CHANGE THIS
        if self.model and hasattr(self.model, 'print_log_queue') and callable(self.model.print_log_queue):
            self.model.print_log_queue(status=status)

    def reset(self):
        super().reset()
        self.model.reset()


if __name__ == '__main__':
    from config import CONFIG
    from src.model.ddpgModel.ddpgModel import DDPGModel

    conf = Config(standard_key_list=TargetAgent.key_list)
    conf.load_config(path=CONFIG + '/ddpgAgentTestConfig.json')

    ddog_con = Config(standard_key_list=DDPGModel.key_list)
    ddog_con.load_config(path=CONFIG + '/targetModelTestConfig.json')

    ddpg = DDPGModel(config=ddog_con)

    a = TargetAgent(config=conf,
                    real_env=2,
                    cyber_env=1,
                    model=ddpg)
