import argparse
import collections
import contextlib
import copy
import importlib
import logging
import os
import sys
import warnings
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor



def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))