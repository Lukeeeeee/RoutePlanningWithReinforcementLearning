import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers import variance_scaling_initializer as contrib_W_init


class NetworkCreator(object):
    act_dict = {
        'LINEAR': tf.identity,
        'RELU': tf.nn.relu,
        'LEAKY_RELU': tf.nn.leaky_relu,
        'SIGMOID': tf.nn.sigmoid,
        'SOFTMAX': tf.nn.softmax,
        'IDENTITY': tf.identity,
        'TANH': tf.nn.tanh,
        'ELU': tf.nn.elu
    }

    def __init__(self):
        pass

    @staticmethod
    def create_network(input, network_config, net_name=None, input_norm=None, output_norm=None,
                       output_low=None, output_high=None, reuse=False):
        # network_config should be a list consist of dict
        # input_norm, output_norm is a list consist of two tensorflow placeholder

        net = tl.layers.InputLayer(inputs=input,
                                   name=net_name + '_INPUT')

        if input_norm:
            net = tl.layers.LambdaLayer(prev_layer=net,
                                        fn=lambda x: (x - input_norm[0]) / input_norm[1])
        last_layer_act = None
        for layer_config in network_config:
            if layer_config['TYPE'] == 'DENSE':
                if layer_config['B_INIT_VALUE'] == 'None':
                    b_init = None
                else:
                    b_init = tf.constant_initializer(value=layer_config['B_INIT_VALUE'])

                net = tl.layers.DenseLayer(prev_layer=net,
                                           n_units=layer_config['N_UNITS'],
                                           act=NetworkCreator.act_dict[layer_config['ACT']],
                                           name=net_name + '_' + layer_config['NAME'],
                                           W_init=contrib_W_init(),
                                           b_init=b_init
                                           )
                last_layer_act = layer_config['ACT']
        if output_norm:
            net = tl.layers.LambdaLayer(prev_layer=net,
                                        fn=lambda x: (x * output_norm[1]) + output_norm[1],
                                        name=net_name + '_NORM')
        if output_high is not None and output_low is not None:
            if last_layer_act != "IDENTITY":
                raise ValueError('Please set the last layer activation as identity to use output scale')
            net = tl.layers.LambdaLayer(prev_layer=net,
                                        fn=lambda x: tf.nn.tanh(x),
                                        name=net_name + '_TANH')
            net = tl.layers.LambdaLayer(prev_layer=net,
                                        fn=lambda x: (x + 1.0) / 2.0 * (output_high - output_low) + output_low,
                                        name=net_name + '_NORM_AFTER_TANH')

            # TODO ADD MORE SUPPORT FOR DIFFERENT LAYER
        return net, net.outputs, net.all_params
