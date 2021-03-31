import json

def save_json(to_save, path):
    with open(path, 'w') as f:
        json.dump(to_save, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_max_input_size(max_power, input_padding):
    input_size = max_power + (2 if input_padding=='pad' else 0)
    return input_size

def get_max_decode_size(max_power):
    return 3*max_power

