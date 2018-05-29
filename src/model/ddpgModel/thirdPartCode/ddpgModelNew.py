from baselines.ddpg.ddpg import DDPG as baseline_ddpg
from src.model.ddpgModel.thirdPartCode.openAIBaselinesModel import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from config.key import CONFIG_KEY
from src.config.config import Config
import tensorflow as tf
from src.model.tensorflowBasedModel import TensorflowBasedModel


class DDPGModelNew(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/ddpgModelKey.json')

    def __init__(self, config, action_bound, obs_bound):
        super(DDPGModelNew, self).__init__(config=config)
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
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                                sigma=float(stddev) * np.ones(nb_actions))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
            self.action_noise = action_noise
            self.para_noise = param_noise

        actor = Actor(nb_actions=self.config.config_dict['ACTION_SPACE'][0],
                      layer_norm=self.config.config_dict['LAYER_NORM_FLAG'],
                      net_config=self.config.config_dict['ACTOR_LAYER_CONFIG'],
                      action_low=action_bound[0],
                      action_high=action_bound[1])
        critic = Critic(net_config=self.config.config_dict['CRITIC_LAYER_CONFIG'])

        self.real_data_memory = Memory(limit=int(1e5),
                                       action_shape=self.config.config_dict['ACTION_SPACE'],
                                       observation_shape=self.config.config_dict['STATE_SPACE'])
        self.simulation_data_memory = Memory(limit=int(1e5),
                                             action_shape=self.config.config_dict['ACTION_SPACE'],
                                             observation_shape=self.config.config_dict['STATE_SPACE'])
        # TODO deal with obs range
        self.ddpg_model = baseline_ddpg(actor=actor,
                                        critic=critic,
                                        memory=self.real_data_memory,
                                        observation_shape=self.config.config_dict['STATE_SPACE'],
                                        action_shape=self.config.config_dict['ACTION_SPACE'],
                                        param_noise=self.para_noise,
                                        action_noise=self.action_noise,
                                        gamma=self.config.config_dict['GAMMA'],
                                        tau=self.config.config_dict['TAU'],
                                        action_range=action_bound,
                                        return_range=(-np.inf, np.inf),
                                        normalize_observations=False,
                                        actor_lr=self.config.config_dict['ACTOR_LEARNING_RATE'],
                                        critic_lr=self.config.config_dict['CRITIC_LEARNING_RATE'],
                                        critic_l2_reg=self.config.config_dict['CRITIC_L2_REG'],
                                        batch_size=self.config.config_dict['BATCH_SIZE'],
                                        observation_range=(-np.inf, np.inf))
        self.ddpg_model.sess = tf.get_default_session()

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.var_list = []
        for var in var_list:
            # TODO THIS MAY LEAD TO SOME BUGS IN THE FUTURE
            if 'actor' in var.name or 'critic' in var.name or 'obs' in var.name:
                self.var_list.append(var)
        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)
        self._env_status = None

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

    @property
    def memory(self):
        return self.ddpg_model.memory

    def update(self):
        # TODO CHECK THIS API
        critic_loss = 0.0
        actor_loss = 0.0
        critic_loss, actor_loss = self.ddpg_model.train()

        self.log_queue.put({self.name + '_ACTOR': actor_loss, self.name + '_CRITIC': critic_loss})
        return critic_loss, actor_loss

    def predict(self, obs):
        if self.status == self.status_key['TEST']:
            action, Q = self.ddpg_model.pi(obs=obs, apply_noise=False, compute_Q=True)
        else:
            action, Q = self.ddpg_model.pi(obs=obs, apply_noise=False, compute_Q=True)
        return action

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
        super().init()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        self.memory.append(obs0=state,
                           obs1=next_state,
                           action=action,
                           reward=reward,
                           terminal1=done)


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
