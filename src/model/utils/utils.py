import numpy as np


def squeeze_array(data, dim=2):
    res = np.squeeze(np.array(data))

    while len(res.shape) < dim:
        res = np.expand_dims(res, 0)
    return res
