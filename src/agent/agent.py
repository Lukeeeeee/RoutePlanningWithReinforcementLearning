from src.core import Basic
import numpy as np
from src.util.sampler.sampler import Sampler


class Agent(Basic):
    key_list = []

    def __init__(self, config, env, model, sampler=Sampler()):
        super(Agent, self).__init__(config)
        self.env = env
        self.model = model
        self._env_step_count = 0
        self.sampler = sampler

    @property
    def env_sample_count(self):
        return self._env_step_count

    @env_sample_count.setter
    def env_sample_count(self, new_value):
        self._env_step_count = new_value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_value):
        if new_value != Basic.status_key['TRAIN'] and new_value != Basic.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)

        if self._status == new_value:
            return
        self._status = new_value
        self.model.status = new_value

    def predict(self, state, *arg, **kwargs):
        return self.model.predict(state)

    def sample(self, env, sample_count, store_flag=False, agent_print_log_flag=False, reset_Flag=True):

        return self.sampler.sample(agent=self,
                                   env=env,
                                   sample_count=sample_count,
                                   store_flag=store_flag,
                                   agent_print_log_flag=agent_print_log_flag,
                                   reset_Flag=reset_Flag)

    def print_log_queue(self, status):
        self.status = status
        reward_list = []
        while self.log_queue.qsize() > 0:
            reward_list.append(self.log_queue.get()[self.name + '_SAMPLE_REWARD'])

        reward_list = np.array(reward_list)
        sum = np.sum(reward_list)
        mean = np.mean(reward_list)
        std = np.mean(reward_list)
        print("%s Reward: Sum: %f Average %f Std %f" %
              (self.name, sum, mean, std))
        self.log_file_content.append({'INDEX': self.log_print_count,
                                      'REWARD_SUM': sum,
                                      'REWARD_MEAN': mean,
                                      'REWARD_STD': std,
                                      'SAMPLE_COUNT': self.env_sample_count})
        self.log_print_count += 1
        # TODO HOW TO ELEGANT CHANGE THIS
        if self.model and hasattr(self.model, 'print_log_queue') and callable(self.model.print_log_queue):
            self.model.print_log_queue(status=status)

    # def evaluate(self, path_nums, horizon):
    #     return self.sample(path_nums=path_nums, horizon=horizon, store_flag=False)

    def init(self):
        print("%s init finished" % type(self).__name__)

    def store_one_sample(self, *args, **kwargs):
        pass

    def update(self):
        pass

    def reset(self):
        pass
