# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Utility functions."""

import numpy as np
import os
import pickle

import dnnlib
import dnnlib.tflib.tfutil as tfutil

BASE_PATH = os.path.dirname(__file__)

#----------------------------------------------------------------------------

def init_tf(random_seed=1234):
    """Initialize TF."""
    print('Initializing TensorFlow...\n')
    np.random.seed(random_seed)
    tfutil.init_tf({'graph_options.place_pruned_graph': True,
                    'gpu_options.allow_growth': True})

#----------------------------------------------------------------------------

def initialize_stylegan():
    """Load StyleGAN network pickle."""
    print('Initializing StyleGAN...\n')
    
    network_path = '/mnt/lustre/users/cgovender/results/baseline/tables/00000-sgan-tables_128-2gpu-mixing-regularization-mix90-stylebased-8/network-snapshot-002364.pkl'#stylegan.pkl trained network
    with open(network_path, "rb") as f:
        _, _, Gs = pickle.load(f) 
    return Gs

#----------------------------------------------------------------------------

def initialize_feature_extractor():
    """Load VGG-16 network pickle (returns features from FC layer with shape=(n, 4096)).""" 
    print('Initializing VGG-16 model...')
   
    vgg16 ='/mnt/lustre/users/cgovender/vgg16.pkl'
    with open(vgg16, "rb") as f:
        _, _ , net = pickle.load(f)
    return net
#----------------------------------------------------------------------------
