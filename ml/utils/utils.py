import copy

def _flatten_dict_of_dict(d, result, prefix, delimiter='/'):
    for key, val in d.items():
        if isinstance(val, dict):
            _flatten_dict_of_dict(val, result, prefix + key + delimiter, delimiter)
        else:
            result[prefix + key] = val

def flatten_dict_of_dict(d, delimiter='/'):
    c = copy.deepcopy(d)
    result = {}
    _flatten_dict_of_dict(c, result, '', delimiter=delimiter)
    return result

def unflatten_dict(d, delimiter='/'):
    c = copy.deepcopy(d)
    result = {}
    for key, val in c.items():
        split = key.split(delimiter)
        cur_node = result
        for split_key in split[:-1]:
            if key not in cur_node:
                cur_node[split_key] = {}
            cur_node = cur_node[split_key]
        cur_node[split[-1]] = val
    return result