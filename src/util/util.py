class SamplerData(object):
    def __init__(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def reset(self):
        self.state_set = []
        self.action_set = []
        self.reward_set = []
        self.done_set = []
        self.new_state_set = []
        self.cumulative_reward = 0.0
        self.step_count_per_episode = 0

    def append(self, state, action, new_state, done, reward):
        self.state_set.append(state)
        self.new_state_set.append(new_state)
        self.reward_set.append(reward)
        self.done_set.append(done)
        self.action_set.append(action)
        self.cumulative_reward += reward

    def __iter__(self):
        for state, new_state, action, reward, done in zip(self.state_set, self.new_state_set, self.action_set,
                                                          self.reward_set, self.done_set):
            yield {
                'STATE': state,
                'NEW_STATE': new_state,
                'ACTION': action,
                'REWARD': reward,
                'DONE': done
            }
