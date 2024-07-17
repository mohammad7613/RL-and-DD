# # # # # Step 0: Import Required Modules # # # # #

import numpy as np
import matplotlib.pyplot as plt

from SimpleGraphVis import *

import pandas as pd

import time

from scipy.io import loadmat

# # # # # Step 1: Load and Generate the Data # # # # #

GC_vals_1_a = np.load("Graph_Sample_1.npy")

raw_locs = np.array([[330, 0],
                     [180, 200],
                     [480, 200],
                     [330, 300],
                     [130, 375],
                     [530, 375],
                     [255, 450],
                     [405, 450],
                     [330, 600],
                     [180, 750],
                     [480, 750],
                     [330, 900],
                     [225, 1000],
                     [435, 1000]])

zero_loc = np.array([330, 500])
locs =  zero_loc - raw_locs

# # # # # Step 2: Define Graph and Pass the Data # # # # #

A = (GC_vals_1_a - np.min(GC_vals_1_a)) / (np.max(GC_vals_1_a) - np.min(GC_vals_1_a))
G = DynamicGraph_NDW(A)

G.get_coords(locs)

t_ = np.arange(-0.4, 1.2, 1.6 / len(A))

G.get_times(t_)

# # # # # Step 3: Visualize The Graph # # # # #

G.draw_dyn_graph(fig_size = [8, 8], delay = 0.1, VisThresh = 0.2)