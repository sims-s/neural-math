import numpy as np


CONFIGS = {
    'baseline' : {
        'exp_vs_num_prob': np.array([
            [1.,0.],
            [.5, .5],
            [.5,.5],
            [0.,1.],
            ]),
        'functions' : ['*', '+', '—'],
        'in_sample_vals' : np.arange(100),
        'oos_vals' : np.arange(100,110),
    },
    'baseline+-' : {
        'exp_vs_num_prob': np.array([
            [1.,0.],
            [.5, .5],
            [.5,.5],
            [0.,1.],
            ]),
        'functions' : ['+', '—'],
        'in_sample_vals' : np.arange(100),
        'oos_vals' : np.arange(100,110),
    },
    'addition' : {
        'exp_vs_num_prob' : np.array([
            [1., 0.],
            [.5, .5],
            [.5, .5],
            [0., 1.]
        ]),
        'functions' : ['+'],
        'in_sample_vals' : np.arange(-99, 100),
        'oos_vals' : np.concatenate([np.arange(-109, -100), np.arange(100, 110)]),

    }
}

# if not specify full vals --> concat in/oos ones
for k, v in CONFIGS.items():
    if not 'full_vals' in v:
        CONFIGS[k]['full_vals'] = np.concatenate((v['in_sample_vals'], v['oos_vals']))