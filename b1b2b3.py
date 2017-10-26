# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class b1b2b3:
    def __init__(self):
        atoms = np.zeros(24)
        rids = np.zeros(24)

        num_atom = 0
        num_rid = 0