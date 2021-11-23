import os
import argparse
import yaml
from easydict import EasyDict

from ml.utils.utils import unflatten_dict, flatten_dict_of_dict

''' Notes
- to add a config parameter, simply add it to default_config.yaml
'''

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'default_config.yaml'))
# DEFAULT_CONFIG = None  # updated below
# DEFAULT_CONFIG_KEYS = None


def read_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

DEFAULT_CONFIG = read_yaml(DEFAULT_CONFIG_PATH)
DEFAULT_CONFIG_KEYS = set(flatten_dict_of_dict(DEFAULT_CONFIG))

def get_command_arguments():
    parser = argparse.ArgumentParser('arguments for training')

    parser.add_argument('config_path', type=str, help='config file path')

    for key, val in DEFAULT_CONFIG.items():
        argument = f"--{'_'.join(key.split('/'))}"
        parser.add_argument(argument, type=type(val))

    opt = parser.parse_args()

    return opt

def read_config(path):
    contents = read_yaml(path)
    keys = set(flatten_dict_of_dict(contents))
    unseen = keys - DEFAULT_CONFIG_KEYS

    if len(unseen) > 0:
        raise Exception(f'Unseen Config Parameters: {unseen}')

    return EasyDict(contents)

def update_config_opt(config, opt):
    opt_to_key = {'_'.join(key.split('/')): key  for key in DEFAULT_CONFIG_KEYS}

    for arg_name, val in vars(opt).items():
        if arg_name == 'config_path':
            config.config_path = val
            continue
        if val is not None:
            key = opt_to_key[arg_name]
            key_split = key.split('/')
            cur_node = config
            
            for split in key_split[:-1]:
                if split not in cur_node:
                    cur_node[split] = {}
                cur_node = cur_node[split]

            cur_node[split[-1]] = val

def process_command_line_and_get_config():
    # returns: easydict (ie dict where keys can be treated as attributes)
    opt = get_command_arguments()
    config = read_config(opt.config_path)
    update_config_opt(config, opt)
    return config

def check_required_config_params(config, required_params):
    missing_required = set(required_params) - set(config.keys())
    if len(missing_required) > 0:
        raise ValueError(f'Missing params in agent config: {missing_required}')

    
    


