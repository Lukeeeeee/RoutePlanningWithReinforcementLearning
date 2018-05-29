from baselines.ddpg.memory import Memory

import numpy as np


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class MemoryForDQN(Memory):
    def __init__(self, limit, action_shape, observation_shape, action_list):
        super().__init__(limit, action_shape, observation_shape)
        self.action_list = action_list
        self.max_count_per_action = max(1, int(limit / len(action_list)))  ###newly modified

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        ###examine if the action belongs to action_list
        inlist = False
        for i in range(len(self.action_list)):
            if np.mean((action - self.action_list[i]) ** 2) < 0.01:
                inlist = True
                break
        idx_set = []
        if inlist == True:
            update_action = self.action_list[i]
            for idx in range(self.actions.length):
                if np.mean((self.actions.__getitem__(idx) - update_action) ** 2) < 0.01:
                    idx_set.append(idx)
        if inlist == True and len(idx_set) > self.max_count_per_action:
            ###count number of samples of
            idx = idx_set[np.random.randint(len(idx_set))]

            start = self.observations0.start
            maxlen = self.observations0.maxlen

            self.observations0.data[(start + idx) % maxlen] = obs0
            self.actions.data[(start + idx) % maxlen] = action
            self.rewards.data[(start + idx) % maxlen] = reward
            self.observations1.data[(start + idx) % maxlen] = obs1
            self.terminals1.data[(start + idx) % maxlen] = terminal1

        else:
            self.observations0.append(obs0)
            self.actions.append(action)
            self.rewards.append(reward)
            self.observations1.append(obs1)
            self.terminals1.append(terminal1)
        print("Current DQN memory=")
        for i in range(self.nb_entries):
            print('Entry ', i, ' = ', self.observations0.get_batch(i), self.actions.get_batch(i),
                  self.rewards.get_batch(i))

    ###sample the one with the best reward
    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries, size=batch_size)
        print("Sample really worked")

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result
