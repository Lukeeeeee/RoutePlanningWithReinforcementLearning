import numpy as np

from baselines.ddpg.memory import Memory
from src.config.config import Config
import tensorflow as tf
import itertools
from src.model.tensorflowBasedModel import TensorflowBasedModel
from src.model.utils.networkCreator import NetworkCreator
from src.model.utils.memory_dqn import MemoryForDQN
import tensorflow.contrib as tfcontrib
from config.key import CONFIG_KEY
import easy_tf_log
from src.model.ddpgModel.ddpgModel import UONoise


class DQNModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/dqnModelKey.json')

    def __init__(self, config, action_bound):
        super(DQNModel, self).__init__(config=config)
        self.proposed_action_list = []
        self.action_bound = action_bound
        action_list = []
        for i in range(len(action_bound[0])):
            low = action_bound[0][i]
            high = action_bound[1][i]
            action_list.append(np.arange(start=low,
                                         stop=high + 0.01,
                                         step=(high - low) / (self.config.config_dict['ACTION_SPLIT_COUNT'] - 1)))
        self.action_step = (action_bound[1] - action_bound[0]) / (self.config.config_dict['ACTION_SPLIT_COUNT'] - 1)
        self.action_iterator = np.asarray(list(itertools.product(*action_list)))
        print("self.action_iterator=", self.action_iterator, )
        # self.action_sample_list = []
        # for sample in self.action_iterator:
        #     self.action_sample_list
        self.noise = UONoise()
        self.noise_scale = 0.1 * (action_bound[1][i] - action_bound[0][i])
        self.reward_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']), dtype=tf.float32)
        self.next_state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']),
                                               dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None] + list(self.config.config_dict['ACTION_SPACE']),
                                           dtype=tf.float32)
        self.done_input = tf.placeholder(shape=[None, 1], dtype=tf.bool)
        self.target_q_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.input = tf.concat([self.state_input, self.action_input], axis=-1)
        self.done = tf.cast(self.done_input, dtype=tf.float32)
        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self.q_net, self.q_output, self.trainable_var_list = NetworkCreator.create_network(input=self.input,
                                                                                               network_config=
                                                                                               self.config.config_dict[
                                                                                                   'NET_CONFIG'],
                                                                                               net_name='Q'
                                                                                               )
            self.target_q_net, self.target_q_output, self.trainable_target_var_list = NetworkCreator.create_network(
                input=self.input,
                network_config=
                self.config.config_dict[
                    'NET_CONFIG'],
                net_name='TARGET_Q')
        self.predict_q_value = (1. - self.done) * self.config.config_dict['DISCOUNT'] * self.target_q_input \
                               + self.reward_input

        self.loss, self.optimizer, self.optimize = self.create_training_method()
        self.update_target_q_op = self.create_target_q_update()
        self.memory = MemoryForDQN(limit=int(self.config.config_dict['MEMORY_SIZE']),
                                   action_shape=self.config.config_dict['ACTION_SPACE'],
                                   observation_shape=self.config.config_dict['STATE_SPACE'],
                                   action_list=self.action_iterator)

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.config_dict['NAME'])

        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)
        self.sess = tf.get_default_session()

    def update(self):
        average_loss = 0.0
        for i in range(self.config.config_dict['ITERATION_EVER_EPOCH']):
            #            print("memory length=", self.memory.)
            if self.memory.observations0.length < self.config.config_dict['BATCH_SIZE']:
                return
            batch_data = self.memory.sample(batch_size=self.config.config_dict['BATCH_SIZE'])
            target_q_value_list = []
            for state in batch_data['obs1']:
                _, target_q_value = self.predict_target(sess=self.sess, new_obs=state)
                target_q_value_list.append(target_q_value)
            re = self.sess.run(fetches=[self.loss, self.optimize],
                               feed_dict={
                                   self.reward_input: batch_data['rewards'],
                                   self.action_input: batch_data['actions'],
                                   self.state_input: batch_data['obs0'],
                                   self.done_input: batch_data['terminals1'],
                                   self.target_q_input: target_q_value_list
                               })
            average_loss += re[0]
        average_loss /= self.config.config_dict['ITERATION_EVER_EPOCH']
        self.log_queue.put({self.name + '_LOSS': average_loss})
        easy_tf_log.tflog(key=self.name + 'TRAIN_LOSS', value=average_loss)
        # TODO POLICY FOR UPDATE DQN TARGET
        self.sess.run(self.update_target_q_op)

    def predict(self, sess, state):
        res, _ = self._predict_action(sess=sess, state=state, q_value_tensor=self.q_output)
        return res

    def predict_target(self, sess, new_obs):
        res, q = self._predict_action(sess=sess, state=new_obs, q_value_tensor=self.target_q_output)
        return res, q

    def init(self):
        sess = tf.get_default_session()
        sess.run(self.variables_initializer)
        super().init()

    def _predict_action(self, sess, state, q_value_tensor):
        if len(state.shape) < 2:
            state_m = state.reshape([1, -1])
        else:
            state_m = state

        actions = []
        q_value_max = []
        for i in range(len(state_m)):
            tmpstates = np.asarray([state_m[i, :] for j in range(len(self.action_iterator))])
            res = sess.run(fetches=[q_value_tensor],
                           feed_dict={
                               self.state_input: tmpstates,
                               self.action_input: self.action_iterator
                           })
            Qvalue = (res[0]).reshape([-1, ])
            Qrange = max(Qvalue) - min(Qvalue) + 1e-9
            Qvalue = 0.9 * (Qvalue - min(
                Qvalue)) + 0.1 * np.random.rand() * Qrange
            actions.append(self.action_iterator[np.argmax(Qvalue), :])
            q_value_max.append(np.max(Qvalue))

        actions = np.asarray(actions)

        if len(actions) < 2:
            actions = actions[0]
        q_value_max = np.asarray(q_value_max)
        return actions, q_value_max

    def create_training_method(self):
        l1_l2 = tfcontrib.layers.l1_l2_regularizer()
        loss = tf.reduce_sum((self.predict_q_value - self.q_output) ** 2) + \
               tfcontrib.layers.apply_regularization(l1_l2, weights_list=self.trainable_var_list)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize_op = optimizer.minimize(loss=loss, var_list=self.trainable_var_list)
        return loss, optimizer, optimize_op

    def create_target_q_update(self):
        op = []
        for var, target_var in zip(self.trainable_var_list, self.trainable_target_var_list):
            ref_val = self.config.config_dict['DECAY'] * target_var + (1.0 - self.config.config_dict['DECAY']) * var
            op.append(tf.assign(target_var, ref_val))
        return op

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        self.memory.append(obs0=state,
                           obs1=next_state,
                           action=action,
                           reward=reward,
                           terminal1=done)

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Critic loss %f: " %
                  (self.name, log[self.name + '_LOSS']))
            log['INDEX'] = self.log_print_count
            self.log_file_content.append(log)
            self.log_print_count += 1
