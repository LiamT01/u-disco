import random

import numpy as np
import torch


def set_device(seed: int) -> torch.device:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Turn on TensorFloat32 (speeds up large model training substantially)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device
