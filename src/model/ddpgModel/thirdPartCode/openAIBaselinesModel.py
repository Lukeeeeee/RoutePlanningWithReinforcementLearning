import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.layers import variance_scaling_initializer as contrib_W_init

act_dict = {

    'RELU': tf.nn.relu,
    'LEAKY_RELU': tf.nn.leaky_relu,
    'SIGMOID': tf.nn.sigmoid,
    'SOFTMAX': tf.nn.softmax,
    'IDENTITY': tf.identity,
    'TANH': tf.nn.tanh,
    'ELU': tf.nn.elu
}


class Model(object):
    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, net_config, nb_actions, action_low, action_high, scope='', name='actor', layer_norm=0):
        super(Actor, self).__init__(name=name, scope=scope)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.net_config = net_config
        self.action_low = action_low
        self.action_high = action_high

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            last_act = None
            for config in self.net_config:
                use_bias = not (config['B_INIT_VALUE']) == 'None'
                if config['TYPE'] == 'DENSE':
                    x = tf.layers.dense(x,
                                        config['N_UNITS'],
                                        kernel_initializer=contrib_W_init(),
                                        bias_initializer=tf.constant_initializer(value=config['B_INIT_VALUE']),
                                        use_bias=use_bias)
                else:
                    raise NotImplementedError("Not support this type layer: %s" % config['TYPE'])
                if self.layer_norm == 1 and config['NAME'] != 'OUTPUT':
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                if act_dict[config['ACT']] is None:
                    raise NotImplementedError("Not support this type activation function: %s" % config['ACT'])
                else:
                    x = act_dict[config['ACT']](x)
                    last_act = config['ACT']
            if last_act == "TANH":
                x = (x + 1.0) / 2.0 * (self.action_high - self.action_low) + self.action_low
            elif last_act == 'SIGMOID':
                x = x * (self.action_high - self.action_low) + self.action_low
            else:
                raise ValueError('Change last act to tanh or sigmoid')
        return x


class Critic(Model):
    def __init__(self, net_config, scope, name='critic', layer_norm=0):
        super(Critic, self).__init__(name=name, scope=scope)
        self.layer_norm = layer_norm
        self.net_config = net_config

    def __call__(self, obs, action, reuse=False):

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            y = action
            count = 0
            for config in self.net_config:
                if count == 0:
                    x = tf.layers.dense(x,
                                        config['N_UNITS'],
                                        kernel_initializer=contrib_W_init(),
                                        bias_initializer=tf.constant_initializer(value=config['B_INIT_VALUE']))
                    y = tf.layers.dense(y,
                                        config['N_UNITS'],
                                        kernel_initializer=contrib_W_init(),
                                        bias_initializer=tf.constant_initializer(value=config['B_INIT_VALUE']))
                    x = tf.concat([x, y], axis=1)
                    count += 1
                else:
                    if config['TYPE'] == 'DENSE':
                        x = tf.layers.dense(x,
                                            config['N_UNITS'],
                                            kernel_initializer=contrib_W_init(),
                                            bias_initializer=tf.constant_initializer(value=config['B_INIT_VALUE']))
                    else:
                        raise NotImplementedError("Not support this type layer: %s" % config['TYPE'])
                    if self.layer_norm == 1 and config['NAME'] != 'OUTPUT':
                        x = tc.layers.layer_norm(x, center=True, scale=True)
                    if act_dict[config['ACT']] is None:
                        raise NotImplementedError("Not support this type activation function: %s" % config['ACT'])
                    else:
                        x = act_dict[config['ACT']](x)
                    count += 1
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
