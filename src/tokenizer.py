import numpy as np
from utils import drop_from_iterable


class Tokenizer():
    def __init__(self, base, problem_specific_tokens=None):
        if not problem_specific_tokens:
            raise ValueError('Expected to find some problem specific tokens; must include EoS ("[EOS]") and SoS ("[SOS]"); SoS MUST BE LAST token')
        if not problem_specific_tokens[-1] == "[SOS]":
            raise ValueError(f"Expected last problem specific token to be Start of Sequence, '[SOS]'; got {problem_specific_tokens[-1]}")
                
        token_mapper = {str(i):i for i in range(base)}
        # Make sure start of sequence token is always the last token in the tokenizer
        # (Needed in generation_utils.decode)
        if problem_specific_tokens:
            token_mapper.update({token:base + i for i, token in enumerate(problem_specific_tokens)})        
        self.token_mapper = token_mapper
        self.inverse = {v: k for k,v in self.token_mapper.items()}

        self.pad_id = self.encode(['[PAD]'])[0]
        self.sos_id = self.encode(['[SOS]'])[0]
        self.eos_id = self.encode(['[EOS]'])[0]
        
    def encode_single(self, seq):
        return [self.token_mapper[str(c)] for c in seq]
    
    def decode_single(self, seq, decode_special=False):
        full = ' '.join([self.inverse[int(t)] for t in seq])
        if decode_special:
            return full
        return self.drop_special(full)

    def drop_special(self, string):
        return drop_from_iterable(string, ['[EOS]', '[PAD]', '[SOS]'])


    def encode(self, seq_or_batch):
        # Encoding a batch
        if hasattr(seq_or_batch[0], '__iter__') and not isinstance(seq_or_batch[0], str):
            max_len = max([len(y) for y in seq_or_batch])
            for i in range(len(seq_or_batch)):
                seq_or_batch[i] = seq_or_batch[i] + ['[PAD]']*(max_len - len(seq_or_batch[i]))
            np_special_case = isinstance(seq_or_batch, np.ndarray)
            if np_special_case:
                return np.array([self.encode_single(seq) for seq in seq_or_batch])
            else:
                return type(seq_or_batch)([self.encode_single(seq) for seq in seq_or_batch])
        # Encoding a single item
        else:
            return self.encode_single(seq_or_batch)

    def decode(self, seq_or_batch, decode_special=False):
        if hasattr(seq_or_batch[0], '__iter__'):
            np_special_case = isinstance(seq_or_batch, np.ndarray)
            if np_special_case:
                return np.array([self.decode_single(seq, decode_special) for seq in seq_or_batch])
            else:
                return type(seq_or_batch)([self.decode_single(seq) for seq in seq_or_batch])
        else:
            return self.decode_single(seq_or_batch, decode_special)
    
    def __len__(self):
        return len(self.token_mapper)
