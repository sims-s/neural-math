import numpy as np


CONFIGS = {
    'add_subtract_depth_2_256' : {
        'exp_vs_num_prob': np.array([
            [1.,0.],
            [.5, .5],
            [0.,1.],
            ]),
        'functions' : ['+', '—'],
        'in_sample_vals' : np.arange(-256, 257),
        'oos_vals' : np.concatenate([np.arange(-356, -256), np.arange(257,357)]),
        # Use streaming
        'train_samples' : 0,
        'val_samples' : 50_000,
        'oos_samples' : 50_000,
    },
    'pairwise_addition_99' : {
        'exp_vs_num_prob' : np.array([
            [1., 0.],
            [0., 1.]
        ]),
        'functions' : ['+'],
        'in_sample_vals' : np.arange(-99, 100),
        'oos_vals' : np.concatenate([np.arange(-109, -99), np.arange(100, 110)]),
        # train/val/test covers 100% of possible expresssions
        'train_samples' : 35_000,
        'val_samples' : 4601,
        'oos_samples' : 8360
    },
    'pairwise_addition_256' : {
        'exp_vs_num_prob' : np.array([
            [1., 0.],
            [0., 1.]
        ]),
        'functions' : ['+'],
        'in_sample_vals' : np.arange(-256, 257),
        'oos_vals' : np.concatenate([np.arange(-356, -256), np.arange(257,357)]),
        # train/val covers 100% of possible expresssions; test covers 25%
        'train_samples' : 233_169,
        'val_samples' : 30_000,
        'oos_samples' : 35_600
    },
    "radius=19_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-19, 20),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-29, -19), np.arange(20, 30)]),
        "train_samples": 2737,
        "val_samples": 305,
        "oos_samples": 3920
    },
    "radius=29_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-29, 30),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-39, -29), np.arange(30, 40)]),
        "train_samples": 6265,
        "val_samples": 697,
        "oos_samples": 5520
    },
    "radius=39_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-39, 40),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-49, -39), np.arange(40, 50)]),
        "train_samples": 11233,
        "val_samples": 1249,
        "oos_samples": 7120
    },
    "radius=49_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-49, 50),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-59, -49), np.arange(50, 60)]),
        "train_samples": 17641,
        "val_samples": 1961,
        "oos_samples": 8720
    },
    "radius=99_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-99, 100),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-109, -99), np.arange(100, 110)]),
        "train_samples": 71281,
        "val_samples": 7921,
        "oos_samples": 16720
    },
    "radius=128_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-128, 129),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-299, -128), np.arange(129, 300)]),
        "train_samples": 118888,
        "val_samples": 13210,
        "oos_samples": 50000
    },
    "radius=256_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-256, 257),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-399, -256), np.arange(257, 400)]),
        "train_samples": 473704,
        "val_samples": 52634,
        "oos_samples": 50000
    },
    "radius=384_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-384, 385),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-499, -384), np.arange(385, 500)]),
        "train_samples": 1064449,
        "val_samples": 118273,
        "oos_samples": 50000
    },
    "radius=512_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-512, 513),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-699, -512), np.arange(513, 700)]),
        "train_samples": 1891125,
        "val_samples": 210125,
        "oos_samples": 50000
    },
    "radius=768_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-768, 769),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-899, -768), np.arange(769, 900)]),
        "train_samples": 4252264,
        "val_samples": 472474,
        "oos_samples": 50000
    },
    "radius=1024_functions=+,-_depth=1": {
        "exp_vs_num_prob": [
            [
                1.0,
                0.0
            ],
            [
                0.0,
                1.0
            ]
        ],
        "in_sample_vals": np.arange(-1024, 1025),
        "functions": [
            "+",
            "-"
        ],
        "oos_vals": np.concatenate([np.arange(-2999, -1024), np.arange(1025, 3000)]),
        "train_samples": 7557121,
        "val_samples": 839681,
        "oos_samples": 50000
    }
}

# if not specify full vals --> concat in/oos ones
for k, v in CONFIGS.items():
    if not 'full_vals' in v:
        CONFIGS[k]['full_vals'] = np.concatenate((v['in_sample_vals'], v['oos_vals']))