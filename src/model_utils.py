import os
import numpy as np

def get_best_model_file(path):
    files = [f for f in os.listdir(path) if f.endswith('.pt')]
    original_files = files
    files = [f.split('_')[1] for f in files]
    losses = [float('.'.join(f.split('.')[:-1])) for f in files]
    try:
        best_loss_idx = np.argmin(losses)
    except ValueError:
        return None
    return path + original_files[best_loss_idx]