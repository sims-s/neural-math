import numpy as np

class Tokenizer():
    def __init__(self):
        self.token_mapper = {
            '0' : 0,
            '1' : 1,
            'x' : 2, # multiplication
            '.' : 3, # End of decode
            '_' : 4, # Padding
        }
        self.inverse = {v: k for k,v in self.token_mapper.items()}
        
    def encode(self, seq):
        return [self.token_mapper[c] for c in seq]
    
    def decode(self, seq, decode_special=False):
        full = ''.join([self.inverse[int(t)] for t in seq])
        if decode_special:
            return full
        return full.replace('.', '').replace('_', '')
    
    # picks encode/decode and handles batch/single based on types
    def __call__(self, x, decode_special=False):
        # Just encode a single item
        if isinstance(x, str):
            return self.encode(x)
        # Encode a batch
        elif isinstance(x[0], str):
            # handle numpy array as a special case b/c you can't create an instance of an array using np.ndarray(foo)...
            np_special_case = isinstance(x, np.ndarray)
            if np_special_case:
                return np.array([self(seq) for seq in x])
            else:
                return type(x)([self(seq) for seq in x])
        # single item to decode
        elif isinstance(x[0], np.integer):
            return self.decode(x, decode_special)
        # Multipel items to decode, may be list or tensor or who knows
        elif hasattr(x[0], '__iter__'):
            np_special_case = isinstance(x, np.ndarray)
            if np_special_case:
                return np.array([self(seq, decode_special) for seq in x])
            else:
                return type(x)([self(seq, decode_special) for seq in x])
    def __len__(self):
        return len(self.token_mapper)
            