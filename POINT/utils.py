import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int) -> None:
    """set seed in few platforms

    Args:
        seed (int): the seed to set
    """    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)