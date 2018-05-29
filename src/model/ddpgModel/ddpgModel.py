from baselines.ddpg.ddpg import DDPG as baseline_ddpg
from src.model.ddpgModel.thirdPartCode.openAIBaselinesModel import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from config.key import CONFIG_KEY
from src.config.config import Config
import tensorflow as tf
from src.model.tensorflowBasedModel import TensorflowBasedModel
import easy_tf_log


class UONoise(object):
    def __init__(self):
        self.theta = 0.15
        self.sigma = 0.2
        self.state = 0.0

    def __call__(self):
        state = self.state - self.theta * self.state + self.sigma * np.random.randn()
        self.state = state
        return self.state

    def reset(self):
        self.state = 0.0


class DDPGModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/ddpgModelKey.json')

    def __init__(self, config, action_bound, obs_bound):
        super(DDPGModel, self).__init__(config=config)
        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self._init(action_bound)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.var_list = []
        for var in var_list:
            # TODO THIS MAY LEAD TO SOME BUGS IN THE FUTURE
            if self.config.config_dict['NAME'] in var.name:
                self.var_list.append(var)
        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)
        self._env_status = None
        self.update_count = 0.0

    def _init(self, action_bound):
        self.action_noise = None
        self.para_noise = None
        if self.config.config_dict['NOISE_FLAG']:
            nb_actions = self.config.config_dict['ACTION_SPACE']
            noise_type = self.config.config_dict['NOISE_TYPE']
            action_noise = None
            param_noise = None
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                         desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = UONoise()
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
            self.action_noise = action_noise
            self.para_noise = param_noise

        actor = Actor(nb_actions=self.config.config_dict['ACTION_SPACE'][0],
                      layer_norm=self.config.config_dict['LAYER_NORM_FLAG'],
                      net_config=self.config.config_dict['ACTOR_LAYER_CONFIG'],
                      action_low=action_bound[0],
                      action_high=action_bound[1],
                      scope=self.config.config_dict['NAME'])
        critic = Critic(net_config=self.config.config_dict['CRITIC_LAYER_CONFIG'],
                        layer_norm=self.config.config_dict['LAYER_NORM_FLAG'],
                        scope=self.config.config_dict['NAME'])

        self.real_data_memory = Memory(limit=int(self.config.config_dict['MEMORY_SIZE_REAL']),
                                       action_shape=self.config.config_dict['ACTION_SPACE'],
                                       observation_shape=self.config.config_dict['STATE_SPACE'])
        self.simulation_data_memory = Memory(limit=int(self.config.config_dict['MEMORY_SIZE_CYBER']),
                                             action_shape=self.config.config_dict['ACTION_SPACE'],
                                             observation_shape=self.config.config_dict['STATE_SPACE'])
        # TODO deal with obs range

        if self.config.config_dict['CLIP_NORM'] > 0.0:
            clip_norm = self.config.config_dict['CLIP_NORM']
        else:
            clip_norm = None

        self.ddpg_model = baseline_ddpg(actor=actor,
                                        critic=critic,
                                        memory=self.real_data_memory,
                                        observation_shape=self.config.config_dict['STATE_SPACE'],
                                        action_shape=self.config.config_dict['ACTION_SPACE'],
                                        param_noise=self.para_noise,
                                        action_noise=None,
                                        gamma=self.config.config_dict['GAMMA'],
                                        tau=self.config.config_dict['TAU'],
                                        normalize_returns=False,
                                        enable_popart=False,
                                        normalize_observations=bool(self.config.config_dict['STATE_NORMALIZATION']),
                                        batch_size=self.config.config_dict['BATCH_SIZE'],
                                        observation_range=(-np.inf, np.inf),
                                        action_range=action_bound,
                                        return_range=(-np.inf, np.inf),
                                        adaptive_param_noise=False,
                                        adaptive_param_noise_policy_threshold=0.1,
                                        critic_l2_reg=self.config.config_dict['CRITIC_L2_REG'],
                                        actor_lr=self.config.config_dict['ACTOR_LEARNING_RATE'],
                                        critic_lr=self.config.config_dict['CRITIC_LEARNING_RATE'],
                                        clip_norm=clip_norm,
                                        reward_scale=1.0)
        self.ddpg_model.sess = tf.get_default_session()

    @property
    def env_status(self):
        return self._env_status

    @env_status.setter
    def env_status(self, new):
        self._env_status = new
        # TODO change
        if new == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.ddpg_model.memory = self.real_data_memory
        elif new == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self.ddpg_model.memory = self.simulation_data_memory
        else:
            raise KeyError('Environment status did not existed')

    @property
    def memory_length(self):
        return self.memory.observations0.length

    @property
    def current_env_status(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return 'REAL_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return 'CYBER_ENVIRONMENT_STATUS'

    @property
    def memory(self):
        return self.ddpg_model.memory

    def update(self):
        self.update_count += 1
        if self.update_count % 50 == 0:
            self.ddpg_model.adapt_param_noise()
        # TODO CHECK THIS API
        critic_loss, actor_loss = self.ddpg_model.train()
        self.ddpg_model.update_target_net()
        self.log_queue.put({self.name + '_ACTOR': actor_loss, self.name + '_CRITIC': critic_loss})
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_ACTOR_TRAIN_LOSS', value=actor_loss)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_CRITIC_TRAIN_LOSS', value=critic_loss)
        self.compute_grad()
        return {
            'VALUE_FUNCTION_LOSS': critic_loss,
            'CONTROLLER_LOSS': actor_loss
        }

    def predict(self, obs):
        if self.status == self.status_key['TEST']:
            action, Q = self.ddpg_model.pi(obs=obs, apply_noise=False, compute_Q=True)
        else:
            action, Q = self.ddpg_model.pi(obs=obs, apply_noise=False, compute_Q=True)
        return action

    def compute_grad(self):
        batch = self.ddpg_model.memory.sample(batch_size=self.ddpg_model.batch_size)
        if self.ddpg_model.normalize_returns and self.ddpg_model.enable_popart:
            old_mean, old_std, target_Q = self.ddpg_model.sess.run([self.ddpg_model.ret_rms.mean,
                                                                    self.ddpg_model.ret_rms.std,
                                                                    self.ddpg_model.target_Q],
                                                                   feed_dict={
                                                                       self.ddpg_model.obs1: batch['obs1'],
                                                                       self.ddpg_model.rewards: batch['rewards'],
                                                                       self.ddpg_model.terminals1: batch[
                                                                           'terminals1'].astype('float32'),
                                                                   })
            self.ddpg_model.ret_rms.update(target_Q.flatten())
            self.ddpg_model.sess.run(self.ddpg_model.renormalize_Q_outputs_op, feed_dict={
                self.ddpg_model.old_std: np.array([old_std]),
                self.ddpg_model.old_mean: np.array([old_mean]),
            })

        else:
            target_Q = self.ddpg_model.sess.run(self.ddpg_model.target_Q, feed_dict={
                self.ddpg_model.obs1: batch['obs1'],
                self.ddpg_model.rewards: batch['rewards'],
                self.ddpg_model.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.ddpg_model.actor_grads, self.ddpg_model.critic_grads]
        actor_grads, critic_grads = self.ddpg_model.sess.run(ops, feed_dict={
            self.ddpg_model.obs0: batch['obs0'],
            self.ddpg_model.actions: batch['actions'],
            self.ddpg_model.critic_target: target_Q,
        })
        actor_grads_norm = np.sqrt(np.sum(actor_grads ** 2))
        critic_grads_norm = np.sqrt(np.sum(actor_grads_norm ** 2))

        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_ACTOR_GRADS_2_NORM',
                          value=actor_grads_norm)
        easy_tf_log.tflog(key=self.name + '_' + self.current_env_status + '_CRITIC_GRADS_2_NORM',
                          value=critic_grads_norm)

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Actor loss %f Critic loss %f: " %
                  (self.name, log[self.name + '_ACTOR'], log[self.name + '_CRITIC']))
            log['INDEX'] = self.log_print_count
            self.log_file_content.append(log)
            self.log_print_count += 1

    def reset(self):
        self.ddpg_model.reset()

    def init(self):
        sess = tf.get_default_session()
        sess.run(self.variables_initializer)
        self.ddpg_model.actor_optimizer.sync()
        self.ddpg_model.critic_optimizer.sync()
        sess.run(self.ddpg_model.target_init_updates)
        self.env_status = self.config.config_dict['REAL_ENVIRONMENT_STATUS']
        self.status = self.status_key['TRAIN']
        super().init()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        self.memory.append(obs0=state,
                           obs1=next_state,
                           action=action,
                           reward=reward,
                           terminal1=done)

    def q_value(self, state, step=0.0):
        _, Q = self.ddpg_model.pi(obs=state, apply_noise=False, compute_Q=True)
        return Q

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
    from config import CONFIG

    con = Config(standard_key_list=DDPGModel.key_list)
    con.load_config(path=CONFIG + '/targetModelTestConfig.json')

    a = DDPGModel(config=con)
    sess = tf.Session()
    with sess.as_default():
        a.init()
        a.load_snapshot()
        a.save_snapshot()
        a.save_model(path='/home/linsen/.tmp/ddpg-model.ckpt', global_step=1)
        a.load_model(file='/home/linsen/.tmp/ddpg-model.ckpt-1')
