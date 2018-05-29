import json
import os


class Config(object):
    def __init__(self, standard_key_list, config_dict=None):
        self.standard_key_list = standard_key_list

        if config_dict:
            self.config_dict = config_dict
        else:
            self._config_dict = {}

    @property
    def config_dict(self):
        return self._config_dict

    @config_dict.setter
    def config_dict(self, new_value):
        if self.check_config(dict=new_value, key_list=self.standard_key_list) is True:
            for key, val in new_value.items():
                if type(val) is list:
                    new_value[str(key)] = tuple(val)
            self._config_dict = new_value

    def save_config(self, path, name):
        Config.save_to_json(dict=self.config_dict, path=path, file_name=name)

    def load_config(self, path):
        res = Config.load_json(file_path=path)
        self.config_dict = res

    def check_config(self, dict, key_list):
        if Config.check_dict_key(dict=dict, standard_key_list=key_list):
            return True
        else:
            return False

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            res = json.load(f)
            return res

    @staticmethod
    def save_to_json(dict, path, file_name=None):
        if file_name is not None:
            path = os.path.join(path, file_name)
        with open(path, 'w') as f:
            json.dump(obj=dict, fp=f, indent=4)

    @staticmethod
    def check_dict_key(dict, standard_key_list):
        for key in standard_key_list:
            if key not in dict:
                raise IndexError('Missing Key %s' % key)
        return True


if __name__ == '__main__':
    config = {
        'A': 1,
        'B': 2,
        'C': 3
    }
    key_list = ['A', 'B', 'C']
    c = Config(config_dict=config, standard_key_list=key_list)
