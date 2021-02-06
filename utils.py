"""
Collection of sparse utilities
"""
import time
import numpy as np
import torch
from torch.autograd import Variable


def reshape_observations(observations: np.array) -> torch.Tensor:
    """
    Reshape to observations to channel first and normalize the pixel values
    Note: the input size is Grayscaled as used by Deep Mind but the channel is still
          4 because it encodes multiple frames in a single entry
    :param observations: (B, H, W, C) where B is the batch size
    :return: (B, C, H, W) observations
    """
    return Variable(torch.from_numpy(observations.transpose((0, 3, 1, 2))).float() / 255.)


##### TIMING FUNCTIONS FOR PROFILIING #####

class timer(object):
    """
    Can be used as
    with timer() as t:
        do_stuff.....

    It will print the duration of the block
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = time.time() - self.start
        print(self.duration)


def timeit(f):
    """
    Decorator which outputs the duration of a method for each run
    :param f:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return timed
