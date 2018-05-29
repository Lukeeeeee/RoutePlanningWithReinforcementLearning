from src.core import Basic
import tensorflow as tf


class Model(Basic):
    def __init__(self, config, data=None):
        super(Model, self).__init__(config)
        self.config = config
        self.data = data
        self.input = None
        self.delta_state_output = None
        self.snapshot_var = []
        self.save_snapshot_op = []
        self.load_snapshot_op = []

    def create_training_method(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def eval_tensor(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def test(self, *args, **kwargs):
        pass

    def save_model(self, path, *args, **kwargs):
        print("%s saved at %s" % (type(self).__name__, path))

    def load_model(self, path, *args, **kwargs):
        print("%s loaded at %s" % (type(self).__name__, path))

    def save_snapshot(self, *args, **kwargs):
        pass

    def load_snapshot(self, *args, **kwargs):
        pass

    def init(self):
        print("%s init finished" % type(self).__name__)

    def store_one_sample(self, *arg, **kwargs):
        pass

    def return_most_recent_sample(self, *args, **kwargs):
        pass

    def update_scale(self, unscaled_data):
        pass

    def q_value(self, state, step=0):
        pass
