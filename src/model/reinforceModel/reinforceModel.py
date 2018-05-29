from config.key import CONFIG_KEY
from src.config.config import Config
import tensorflow as tf
from src.model.tensorflowBasedModel import TensorflowBasedModel
import easy_tf_log
from src.model.utils.networkCreator import NetworkCreator
import numpy as np
import itertools


class REINFORCEModel(TensorflowBasedModel):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/reinforceModelKey.json')

    def __init__(self, config, action_bound):
        super(REINFORCEModel, self).__init__(config=config)
        self.action_set = []
        self.reward_set = []
        self.state_set = []

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
        self.action_size = len(self.action_iterator)

        self.config.config_dict['NET_CONFIG'][-1]['N_UNITS'] = self.action_size
        assert self.config.config_dict['NET_CONFIG'][-1]['ACT'] == 'SOFTMAX'

        with tf.variable_scope(name_or_scope=self.config.config_dict['NAME']):
            self.state_input = tf.placeholder(shape=[None] + list(self.config.config_dict['STATE_SPACE']),
                                              dtype=tf.float32)

            self.net, self.output, self.trainable_var_list = NetworkCreator.create_network(input=self.state_input,
                                                                                           network_config=
                                                                                           self.config.config_dict[
                                                                                               'NET_CONFIG'],
                                                                                           net_name=
                                                                                           self.config.config_dict[
                                                                                               'NAME'])
            self.advantages = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
            self.loss, self.optimizer, self.optimize_op = self.create_training_method()
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.config_dict['NAME'])

        self.variables_initializer = tf.variables_initializer(var_list=self.var_list)
        self.sess = tf.get_default_session()

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.config.config_dict['DISCOUNT'] + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def update(self, *args, **kwargs):
        episode_length = len(self.state_set)

        discounted_rewards = self.discount_rewards(self.reward_set)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 0.0000001

        update_inputs = np.zeros((episode_length, self.config.config_dict['STATE_SPACE'][0]))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.state_set[i]
            advantages[i][self.action_set[i]] = discounted_rewards[i]

        average_loss = 0.0
        for i in range(self.config.config_dict['ITERATION_EVER_EPOCH']):
            re, _ = self.sess.run(fetches=[self.loss, self.optimize_op],
                                  feed_dict={
                                      self.state_input: update_inputs,
                                      self.advantages: advantages
                                  })
            average_loss += np.sum(re)
        average_loss /= self.config.config_dict['ITERATION_EVER_EPOCH']
        self.log_queue.put({self.name + '_LOSS': average_loss})
        easy_tf_log.tflog(key=self.name + 'TRAIN_LOSS', value=average_loss)
        self.state_set, self.action_set, self.reward_set = [], [], []
        self.print_log_queue(status=self.status_key['TRAIN'])

    def predict(self, sess, state):
        res = sess.run(fetches=[self.output],
                       feed_dict={self.state_input: state})

        res = np.array(res).squeeze()
        res = np.random.choice(self.action_size, 1, p=res)[0]
        return self.action_iterator[res]

    def init(self):
        sess = tf.get_default_session()
        sess.run(self.variables_initializer)
        super().init()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        self.state_set.append(state)
        for i in range(len(self.action_iterator)):
            if np.array(self.action_iterator[i]).tolist() == np.array(action).tolist():
                action = i
        self.action_set.append(action)
        self.reward_set.append(reward)

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Loss %f: " %
                  (self.name, log[self.name + '_LOSS']))
            log['INDEX'] = self.log_print_count
            self.log_file_content.append(log)
            self.log_print_count += 1

    def create_training_method(self, *args, **kwargs):
        loss = tf.keras.backend.categorical_crossentropy(target=self.advantages,
                                                         output=self.output)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.config_dict['LEARNING_RATE'])
        optimize_op = optimizer.minimize(loss=loss, var_list=self.trainable_var_list)
        return loss, optimizer, optimize_op
