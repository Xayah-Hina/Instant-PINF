import numpy as np
import torch
import src.radam as radam
import src.model as mmodel

import os
import math
import time

import taichi as ti

if __name__ == '__main__':
    device = torch.device("cuda")
    ti.init(arch=ti.cuda, device_memory_GB=12.0)
    np.random.seed(0)
