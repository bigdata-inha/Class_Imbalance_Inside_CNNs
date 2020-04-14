import torch
import numpy as np
import random

def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)


