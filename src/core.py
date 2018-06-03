import tensorflow as tf
import time
from log import LOG
import os
from src.config.config import Config
from config.key import CONFIG_KEY
import numpy as np
import random
import queue
import json
import easy_tf_log


class Basic(object):
    key_list = []
    status_key = {'TRAIN': 0, 'TEST': 1}

    def __init__(self, config):
        self.config = config
        self.name = type(self).__name__

        self._train_log_file = self.name + '_train_.log'
        self._test_log_file = self.name + '_test_.log'

        self._train_log_queue = queue.Queue(maxsize=1e10)
        self._test_log_queue = queue.Queue(maxsize=1e10)

        self._train_log_print_count = 0
        self._test_log_print_count = 0

        self._train_log_file_content = []
        self._test_log_file_content = []

        self._status = Basic.status_key['TRAIN']

        self._log_file = None
        self._log_queue = None
        self._log_print_count = None
        self._log_file_content = None

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            content = self.log_queue.get()
            self.log_file_content.append({'INDEX': self.log_print_count, 'LOG': content})
            print(content)
            self.log_print_count += 1

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

    @property
    def log_file(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_queue(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_queue
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_queue
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_file_content(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file_content
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file_content
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_print_count(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_print_count
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_print_count
        raise KeyError('Current Status: %d did not existed' % self._status)

    @log_print_count.setter
    def log_print_count(self, new_val):
        if self._status == Basic.status_key['TRAIN']:
            self._train_log_print_count = new_val
        elif self._status == Basic.status_key['TEST']:
            self._test_log_print_count = new_val
        else:
            raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def current_status(self):
        if self._status == Basic.status_key['TRAIN']:
            return 'TRAIN'
        elif self._status == Basic.status_key['TEST']:
            return 'TEST'


class Logger(object):
    def __init__(self, prefix=None, log=LOG):
        self._log_dir = log + '/' + prefix + '/' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_case2'
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        if os.path.exists(self._log_dir):
            raise FileExistsError('%s path is existed' % self._log_dir)
        self.tf_log = easy_tf_log
        self.tf_log.set_dir(log_dir=self._log_dir + '/tf/')

    @property
    def log_dir(self):
        if os.path.exists(self._log_dir) is False:
            os.makedirs(self._log_dir)
        return self._log_dir

    @property
    def config_file_log_dir(self):
        self._config_file_log_dir = os.path.join(self.log_dir, 'config')
        if os.path.exists(self._config_file_log_dir) is False:
            os.makedirs(self._config_file_log_dir)
        return self._config_file_log_dir

    @property
    def loss_file_log_dir(self):
        self._loss_file_log_dir = os.path.join(self.log_dir, 'loss')
        if os.path.exists(self._loss_file_log_dir) is False:
            os.makedirs(self._loss_file_log_dir)
        return self._loss_file_log_dir

    @property
    def model_file_log_dir(self):
        self._model_file_log_dir = os.path.join(self.log_dir, 'model/')
        if os.path.exists(self._model_file_log_dir) is False:
            os.makedirs(self._model_file_log_dir)
        return self._model_file_log_dir

    def out_to_file(self, file_path, content):
        with open(file_path, 'w') as f:
            # TODO how to modify this part
            for dict_i in content:
                for key, value in dict_i.items():
                    if isinstance(value, np.generic):
                        dict_i[key] = value.item()
            json.dump(content, fp=f, indent=4)


class GamePlayer(object):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/gamePlayerKey.json')

    def __init__(self, config, agent, env, basic_list):
        self.config = config
        self.agent = agent
        self.env = env
        self.basic_list = basic_list

        self.logger = Logger(prefix=self.config.config_dict['GAME_NAME'], log=LOG)

    def set_seed(self, seed=None):
        if seed is None:
            seed = int(self.config.config_dict['SEED'])
        else:
            self.config.config_dict['SEED'] = seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

    def play(self, seed_new=None):
        if seed_new is None:
            self.set_seed()
        else:
            self.set_seed(seed_new)

        if self.config.config_dict['SAVE_CONFIG_FILE_FLAG'] == 1:
            for basic in self.basic_list:
                basic.config.save_config(path=self.logger.config_file_log_dir,
                                         name=basic.name + '.json')
            self.config.save_config(path=self.logger.config_file_log_dir,
                                    name='GamePlayer.json')

        self.agent.init()
        self.env.init()
        info_set = []

        self.agent.env_status = self.agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
        self.agent.status = self.agent.status_key['TRAIN']

        # TODO modify here to control the whole training process
        for i in range(self.config.config_dict['EPOCH']):
            self.agent.env_status = self.agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
            self.agent.status = self.agent.status_key['TRAIN']

            for j in range(self.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                trainer_data = self.agent.sample(env=self.env,
                                                 sample_count=50,
                                                 store_flag=True,
                                                 agent_print_log_flag=True)
                self.agent.train()
                # print("Q table:")
                # print(self.agent.model.q_table)
            self.agent.status = self.agent.status_key['TEST']
            print("Test")
            trainer_data = self.agent.sample(env=self.env,
                                             sample_count=50,
                                             store_flag=False,
                                             agent_print_log_flag=True)
            # print("Q table:")
            # print(self.agent.model.q_table)
        pass

    def print_log_to_file(self):
        for basic in self.basic_list:
            if 'LOG_FLAG' in basic.config.config_dict and basic.config.config_dict['LOG_FLAG'] == 1:
                basic.status = basic.status_key['TRAIN']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)
                basic.status = basic.status_key['TEST']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)

    def save_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.save_model(path=self.logger.model_file_log_dir, global_step=1)

    def load_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.load_model(path=self.logger.model_file_log_dir, global_step=1)


if __name__ == '__main__':
    pass
