import numpy as np

class Tokenizer():
    def __init__(self, base):
        digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        token_mapper = {digits[i] : i for i in range(base)}
        # Make sure start of sequence token is always the last token in the tokenizer
        # (Needed in generation_utils.decode)
        token_mapper.update({
            'x' : base, # multiplication
            '_' : base+1, #padding,
            '.' : base+2, # end of sequence
            '>' : base+3, # start of sequence
        })
        self.token_mapper = token_mapper
        self.inverse = {v: k for k,v in self.token_mapper.items()}
        
    def encode(self, seq):
        return [self.token_mapper[c] for c in seq]
    
    def decode(self, seq, decode_special=False):
        full = ''.join([self.inverse[int(t)] for t in seq])
        if decode_special:
            return full
        return self.drop_special(full)

    def drop_special(self, string):
        return string.replace('.', '').replace('_', '').replace('>', '')
    
    # picks encode/decode and handles batch/single based on types
    def __call__(self, x, decode_special=False):
        # Just encode a single item
        if isinstance(x, str):
            return self.encode(x)
        # Encode a batch
        elif isinstance(x[0], str):
            # Pad everything to be the same length first
            max_len = max([len(y) for y in x])
            for i in range(len(x)):
                x[i] = x[i] + '_'*(max_len - len(x[i]))
            # handle numpy array as a special case b/c you can't create an instance of an array using np.ndarray(foo)...
            np_special_case = isinstance(x, np.ndarray)
            if np_special_case:
                return np.array([self(seq) for seq in x])
            else:
                return type(x)([self(seq) for seq in x])
        # single item to decode
        elif isinstance(x[0], np.integer) or isinstance(x[0], int):
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
            