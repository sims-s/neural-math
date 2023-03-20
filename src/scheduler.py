import numpy as np
from torch.optim.lr_scheduler import LambdaLR


# Adapted from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
class ReduceLROnPlateauWithWarmup(object):
    def __init__(self, optimizer, num_warmup_steps, mode='min', factor=0.1, window_size=100,
                 threshold=1e-4, threshold_mode='rel', patience=100, cooldown=100,
                 min_lr=0, eps=1e-8, verbose=False, nb_steps=-1):

        self.num_warmup_steps = num_warmup_steps

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # self.patience = patience
        self.window_size = window_size
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.patience = patience
        self.patience_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.total_epochs = None
        # self.best = None
        self.hist = None
        # self.num_bad_epochs = None
        # self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        self.hist = []
        self.total_epochs = 0
        self._get_base_lr()

    def update_hist(self, metrics):
        current = float(metrics)
        self.hist.append(current)
        if len(self.hist) > 2*self.window_size:
            self.hist = self.hist[1:]

    
    def step(self, metrics, epoch=None):
        self.update_hist(metrics)

        if self.total_epochs < self.num_warmup_steps:
            self.warmup_step()
        else:
            if self.in_cooldown:
                self.cooldown_counter -=1
            else:
                self.standard_step(metrics, epoch)
        self.total_epochs += 1

    def _set_lr(self, new_lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lr
    def warmup_step(self):
        scale_factor = float(self.total_epochs) / float(max(1, self.num_warmup_steps))

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * scale_factor
    
    def standard_step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        self.last_epoch = epoch

        if self.should_reduce():
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


    
    def _get_base_lr(self):
        base_lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lrs.append(param_group['lr'])
        self.base_lrs = base_lrs

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(self.total_epochs, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def should_reduce(self):
        if len(self.hist) < 2*self.window_size:
            return False
        assert len(self.hist)==2*self.window_size

        older = np.mean(self.hist[:self.window_size])
        newer = np.mean(self.hist[self.window_size:])
        if not self.is_better(newer, older):
            self.patience_counter +=1
            if self.patience_counter > self.patience:
                self.patience_counter = 0
                return not self.is_better(newer, older)
        else:
            self.patience_counter = 0
        return False

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('Inf')
        else:  # mode == 'max':
            self.mode_worse = -float('Inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, nb_steps, min_lr_scale=0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        nb_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    # object not function for saving purposes
    class _LR_Lambda():
        def __init__(self, min_lr_scale):
            self.min_lr_scale = min_lr_scale

        def __call__(self, current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                min_lr_scale, float(nb_steps - current_step) / float(max(1, nb_steps - num_warmup_steps))
            )

    return LambdaLR(optimizer, _LR_Lambda(min_lr_scale), last_epoch)