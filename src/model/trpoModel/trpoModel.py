import numpy as np
import tensorflow as tf
from src.model.trpoModel.trpo.policy import Policy
from src.model.trpoModel.trpo.value_function import NNValueFunction
from src.model.trpoModel.trpo.utils import Logger, Scaler
import src.model.trpoModel.trpo.train as trpo_main
from src.model.tensorflowBasedModel import TensorflowBasedModel
import easy_tf_log
from collections import deque
from src.config.config import Config
from config.key import CONFIG_KEY
from baselines.ddpg.memory import Memory


class TrpoModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/trpoModelKey.json')

    def __init__(self, config, action_bound, obs_bound):
        super().__init__(config=config)
        self.obs_dim = self.config.config_dict['STATE_SPACE']
        self.obs_dim = self.obs_dim[0] + 1
        self.act_dim = self.config.config_dict['ACTION_SPACE'][0]
        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self.scaler = Scaler(self.obs_dim)
            self.val_func = NNValueFunction(self.obs_dim,
                                            hid1_mult=self.config.config_dict['HIDDEN_MULTIPLE'],
                                            name_scope=self.config.config_dict['NAME'])
            self.policy = Policy(self.obs_dim,
                                 self.act_dim,
                                 kl_targ=self.config.config_dict['KL_TARG'],
                                 hid1_mult=self.config.config_dict['HIDDEN_MULTIPLE'],
                                 policy_logvar=self.config.config_dict['POLICY_LOGVAR'],
                                 name_scope=self.config.config_dict['NAME'])

        self._real_trajectories = {'observes': [],
                                   'actions': [],
                                   'rewards': [],
                                   'unscaled_obs': []}

        self._cyber_trajectories = {'observes': [],
                                    'actions': [],
                                    'rewards': [],
                                    'unscaled_obs': []}
        self._real_trajectories_memory = deque(maxlen=self.config.config_dict['EPISODE_REAL_MEMORY_SIZE'])
        self._cyber_trajectories_memory = deque(maxlen=self.config.config_dict['EPISODE_CYBER_MEMORY_SIZE'])
        self._real_step_count = 0.0
        self._cyber_step_count = 0.0
        self.action_low = action_bound[0]
        self.action_high = action_bound[1]
        self._env_status = None

        self.real_data_memory = Memory(limit=10000,
                                       action_shape=self.config.config_dict['ACTION_SPACE'],
                                       observation_shape=self.config.config_dict['STATE_SPACE'])
        self.simulation_data_memory = Memory(limit=10000,
                                             action_shape=self.config.config_dict['ACTION_SPACE'],
                                             observation_shape=self.config.config_dict['STATE_SPACE'])

    @property
    def env_status(self):
        return self._env_status

    @env_status.setter
    def env_status(self, new):
        self._env_status = new
        # TODO change
        if new == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.memory = self.real_data_memory
        elif new == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self.memory = self.simulation_data_memory
        else:
            raise KeyError('Environment status did not existed')

    @property
    def memory_length(self):
        count = 0
        # self._save_trajectories_to_memory(reset_step_count=False)
        for espiode in self.trajectories_memory:
            count += len(espiode['observes'])
        return count

    @property
    def current_env_status(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return 'REAL_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return 'CYBER_ENVIRONMENT_STATUS'

    @property
    def trajectories_memory(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return self._real_trajectories_memory
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return self._cyber_trajectories_memory

    @property
    def trajectories(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return self._real_trajectories
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return self._cyber_trajectories

    @trajectories.setter
    def trajectories(self, new_val):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self._real_trajectories = new_val
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self._cyber_trajectories = new_val

    @property
    def step_count(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return self._real_step_count
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return self._cyber_step_count

    @step_count.setter
    def step_count(self, new_val):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self._real_step_count = new_val
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self._cyber_step_count = new_val

    def update(self):
        # self._save_trajectories_to_memory(reset_step_count=False)
        observes, actions, advantages, disc_sum_rew = self._return_train_data()
        # TODO FIX LOGGER AND UODATE LOG DATA
        loss, entropy, kl, beta, lr_multiplier = self.policy.update(observes=observes,
                                                                    actions=actions,
                                                                    advantages=advantages,
                                                                    logger=None)

        loss_val, exp_var, old_exp_var = self.val_func.fit(x=observes,
                                                           y=disc_sum_rew,
                                                           logger=None)
        res_dict = {
            self.name + '_POLICY_LOSS': loss,
            self.name + '_ENTROPY': entropy,
            self.name + '_KL': kl,
            self.name + '_BETA': beta,
            self.name + '_LR_MULTIPLIER': lr_multiplier,
            self.name + '_VAL_FUNCTION_LOSS': loss_val,
            self.name + '_EXP_VAR': exp_var,
            self.name + '_OLD_EXP_VAR': old_exp_var
        }
        self.log_queue.put(res_dict)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_POLICY_LOSS', value=loss)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_ENTROPY', value=entropy)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_KL', value=kl)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_BETA', value=beta)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_LR_MULTIPLIER', value=lr_multiplier)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_VAL_FUNCTION_LOSS', value=loss_val)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_EXP_VAR', value=exp_var)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_OLD_EXP_VAR', value=old_exp_var)
        return {
            'VALUE_FUNCTION_LOSS': loss_val,
            'CONTROLLER_LOSS': loss
        }

    def predict(self, obs):
        obs = np.reshape(obs, [1, -1])
        obs = np.append(obs, [[self.step_count * self.config.config_dict['INCREMENT_ENV_STEP']]],
                        axis=1)
        scale, offset = self.scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        obs = (obs - offset) * scale
        action = self.policy.sample(np.reshape(obs, [1, -1])).reshape((1, -1)).astype(np.float32)
        action = np.clip(action, a_min=self.action_low, a_max=self.action_high)
        return action

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Policy Loss %f, Entropy %f, Kl %f, Beta %f, Lr multiplier %f, Val function loss %f, "
                  "Exp var %f, Old exp var %f" % (self.name,
                                                  log[self.name + '_POLICY_LOSS'],
                                                  log[self.name + '_ENTROPY'],
                                                  log[self.name + '_KL'],
                                                  log[self.name + '_BETA'],
                                                  log[self.name + '_LR_MULTIPLIER'],
                                                  log[self.name + '_VAL_FUNCTION_LOSS'],
                                                  log[self.name + '_EXP_VAR'],
                                                  log[self.name + '_OLD_EXP_VAR']
                                                  ))
            log['INDEX'] = self.log_print_count
            self.log_print_count += 1
            self.log_file_content.append(log)

    def reset(self):
        self.trajectories = {'observes': [],
                             'actions': [],
                             'rewards': [],
                             'unscaled_obs': []}
        self.step_count = 0

    def init(self):
        self.var_list = self.val_func.var_list + self.policy.var_list
        self.val_func.init()
        self.policy.init()
        self.trajectories = {'observes': [],
                             'actions': [],
                             'rewards': [],
                             'unscaled_obs': []}
        self.step_count = 0
        self.env_status = self.config.config_dict['REAL_ENVIRONMENT_STATUS']
        super().init()
        pass

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        # TODO HOW TO SET AND RESET STEP
        self.memory.append(obs0=state,
                           obs1=next_state,
                           action=action,
                           reward=reward,
                           terminal1=done)
        obs = state.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[self.step_count * self.config.config_dict['INCREMENT_ENV_STEP']]],
                        axis=1)  # add time step feature
        self.trajectories['unscaled_obs'].append(obs)

        scale, offset = self.scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        obs = (obs - offset) * scale  # center and scale observations
        self.trajectories['observes'].append(np.reshape(obs, [-1]))
        self.trajectories['actions'].append(np.reshape(action, [-1]))
        self.trajectories['rewards'].append(reward)
        self.step_count += 1
        if done is True:
            self._save_trajectories_to_memory(reset_step_count=True)

    def _return_train_data(self):
        trajectories = list(self.trajectories_memory)
        trpo_main.add_value(trajectories, val_func=self.val_func)
        trpo_main.add_disc_sum_rew(trajectories=trajectories,
                                   gamma=self.config.config_dict['GAMMA'])
        trpo_main.add_gae(trajectories=trajectories,
                          gamma=self.config.config_dict['GAMMA'],
                          lam=self.config.config_dict['LAM'])
        observes, actions, advantages, disc_sum_rew = trpo_main.build_train_set(trajectories=trajectories)
        self.trajectories_memory.clear()

        return observes[-self.config.config_dict['BATCH_SIZE']:], actions[-self.config.config_dict['BATCH_SIZE']:], \
               advantages[-self.config.config_dict['BATCH_SIZE']:], disc_sum_rew[
                                                                    -self.config.config_dict['BATCH_SIZE']:]

    def _save_trajectories_to_memory(self, reset_step_count=True):
        if len(self.trajectories['observes']) > 0:
            self.update_scale(unscaled_data=np.array(self.trajectories['unscaled_obs']).squeeze())
            if reset_step_count is True:
                self.step_count = 0
            for key, val in self.trajectories.items():
                self.trajectories[key] = np.array(val)
            self.trajectories_memory.append(self.trajectories)
            self.trajectories = {'observes': [],
                                 'actions': [],
                                 'rewards': [],
                                 'unscaled_obs': []}

    def update_scale(self, unscaled_data):
        self.scaler.update(x=unscaled_data)

    def q_value(self, state, step=0):
        return self.val_func.predict(x=np.array(state), step=step * self.config.config_dict['INCREMENT_ENV_STEP'])

    def return_most_recent_sample(self, sample_count, env_status, *args, **kwargs):
        if env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            memory = self.real_data_memory
        elif env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            memory = self.simulation_data_memory
        else:
            raise ValueError('Wrong Environment status')

        length = memory.nb_entries
        enough_flag = True
        if length < sample_count:
            enough_flag = False
        from src.util.sampler.sampler import SamplerData
        sample_data = SamplerData()
        for i in range(max(0, length - sample_count), length):
            sample_data.append(state=memory.observations0[i],
                               action=memory.actions[i],
                               new_state=memory.observations1[i],
                               done=memory.terminals1[i],
                               reward=memory.rewards[i])
        return sample_data, enough_flag


if __name__ == '__main__':

    from src.config.config import Config

    con = Config(standard_key_list=TrpoModel.key_list)
    con.load_config(
        path='/home/dls/CAP/intelligenttrainerframework/config/modelNetworkConfig/targetModelTestConfig.json')
    a = TrpoModel(config=con, action_bound=([-1], [1]), obs_bound=([-1], [1]))
    import tensorflow as tf
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    with sess.as_default():
        a.init()
        for i in range(500):
            a.env_status = a.config.config_dict['CYBER_ENVIRONMENT_STATUS']
            a.store_one_sample(state=np.array([0.1]),
                               next_state=np.array([0.2]),
                               action=a.predict(obs=np.array([0.1, 0.001])),
                               reward=4.1,
                               done=False)
        for i in range(500):
            a.env_status = a.config.config_dict['REAL_ENVIRONMENT_STATUS']
            a.store_one_sample(state=np.array([0.1]),
                               next_state=np.array([0.2]),
                               action=a.predict(obs=np.array([0.1, 0.001])),
                               reward=4.1,
                               done=False)
        a.store_one_sample(state=np.array([0.1]),
                           next_state=np.array([0.2]),
                           action=a.predict(obs=np.array([0.1, 0.001])),
                           reward=4.1,
                           done=True)
        a.update()
        pass
