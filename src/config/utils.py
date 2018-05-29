import numpy as np
import json
import sys
import os


def save_to_json(dict, path, file_name=None):
    if file_name is not None:
        path = os.path.join(path, file_name)
    with open(path, 'r') as f:
        json.dump(obj=dict, fp=f)


def load_json(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
        return res


def check_dict_key(dict, standard_key_list):
    for key in standard_key_list:
        if key not in dict:
            raise IndexError('Missing %s' % key)

    return True
