import numpy as np
from collections import defaultdict

class Scheduler:
    """Updates optimizer's learning rates using the provided scheduling function."""
    def __init__(self, optimizer, schedule):
        self.optimizer = optimizer
        self.schedule = schedule
        self.history = defaultdict(list)

    def step(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.schedule(epoch)
            param_group['lr'] = lr
            self.history[i].append(lr)

    @staticmethod
    def cosine_schedule(epoch, t_max, ampl):
        """Shifted and scaled cosine function."""
        t = epoch % t_max
        return (1 + np.cos(np.pi * t / t_max)) * ampl / 2

    @staticmethod
    def inv_cosine_schedule(epoch, t_max, ampl):
        """A cosine function reflected on the X-axis."""
        return 1 - Scheduler.cosine_schedule(epoch, t_max, ampl)

    @staticmethod
    def one_cycle_schedule(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
        """A combined schedule with two cosine half-waves."""
        pct = epoch / t_max
        if pct < pivot:
            return Scheduler.inv_cosine_schedule(epoch, pivot * t_max, a1)
        return Scheduler.cosine_schedule(epoch - pivot * t_max, (1 - pivot) * t_max, a2)