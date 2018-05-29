from src.core import Basic


class BasicEnv(Basic):
    key_list = []

    def __init__(self, config):
        super(BasicEnv, self).__init__(config=config)
        self.action_space = None
        self.observation_space = None
        self.cost_fn = None
        self.step_count = 0

    def step(self, action):
        self.step_count += 1

    def reset(self):
        return None

    def init(self):
        print("%s init finished" % type(self).__name__)


if __name__ == '__main__':
    a = BasicEnv(config=1)
