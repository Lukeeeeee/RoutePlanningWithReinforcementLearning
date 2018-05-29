import tensorflow as tf
import tensorlayer as tl


class DataNormalizer(object):
    def __init__(self, data_tensor, name):
        self.data = data_tensor

        self.data_shape = self.data.get_shape().as_list()[1:]

        self.mean = tf.placeholder(shape=self.data_shape, dtype=tf.float32)
        self.std = tf.placeholder(shape=self.data_shape, dtype=tf.float32)
        self.name = name
        self.norm_tensor = self.create_normalize()
        self.denorm_tensor = self.create_denormalize()

    def create_normalize(self):
        net = tl.layers.InputLayer(inputs=self.data,
                                   name=self.name + '_NORMALIZE')
        net = tl.layers.LambdaLayer(prev_layer=net,
                                    fn=lambda x: (x - self.mean) / self.std)
        return net.outputs

    def create_denormalize(self):
        net = tl.layers.InputLayer(inputs=self.data,
                                   name=self.name + '_DENORMALIZE')
        net = tl.layers.LambdaLayer(prev_layer=net,
                                    fn=lambda x: (x * self.std) + self.mean)
        return net.outputs
