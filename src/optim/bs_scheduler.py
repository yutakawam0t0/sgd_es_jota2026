import math
from typing import Optional, Callable


class BSScheduler:

    def __init__(self, initial_bs: int, last_epoch: int = -1):
        self.initial_bs = initial_bs
        self.last_epoch = last_epoch
        self.batch_size = initial_bs

        self._initial_step()

    def _initial_step(self):
        """Perform initial step and update batch size"""
        self.step()

    def get_batch_size(self) -> int:
        # Compute batch size using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self.batch_size = self.get_batch_size()

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        return {
            'initial_bs': self.initial_bs,
            'last_epoch': self.last_epoch,
            'batch_size': self.batch_size
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state from a state_dict."""
        self.initial_bs = state_dict['initial_bs']
        self.last_epoch = state_dict['last_epoch']
        self.batch_size = state_dict['batch_size']


class LambdaBS(BSScheduler):
    """Sets the batch size of each parameter group to the initial bs
    times a given function. When last_epoch=-1, sets initial bs as bs.

    Args:
        initial_bs (int): Initial batch size.
        bs_lambda (Callable[[int], int]): A function that takes the current epoch
            as an input and returns the factor to adjust the batch size.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        initial_bs: int,
        bs_lambda: Callable[[int], int],
        last_epoch: int = -1
    ):
        self.bs_lambda = bs_lambda
        super().__init__(initial_bs, last_epoch)

    def get_batch_size(self) -> int:
        return math.ceil(self.initial_bs * self.bs_lambda(self.last_epoch))
