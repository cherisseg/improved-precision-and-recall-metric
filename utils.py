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
    #path = 'https://storage.googleapis.com/stylegan_env/chairs_network-final.pkl' # stylegan.pkl trained network
    #cache_dir=os.path.join(path, '_cache')
    #with open(path) as f:
    #    _, _, Gs = pickle.load(f) 
    #return Gs
    #Google cloud link
    url= 'https://storage.googleapis.com/stylegan_env/lamps_network-final.pkl'
    with dnnlib.util.open_url(url, cache_dir=os.path.join(BASE_PATH, '_cache')) as f:
        _, _, Gs = pickle.load(f) 
    return Gs

#----------------------------------------------------------------------------

def initialize_feature_extractor():
    """Load VGG-16 network pickle (returns features from FC layer with shape=(n, 4096)).""" 
    print('Initializing VGG-16 model...')
    #path = '/mnt/lustre/users/cgovender/vgg16.pkl' # vgg16.pkl
    #cache_dir=os.path.join(path, '_cache')
   # with open(path) as f:
     #   _, _, net = pickle.load(f)
    #return net
    #Google cloud link
    url ='https://storage.googleapis.com/stylegan_env/vgg16.pkl'
    with dnnlib.util.open_url(url, cache_dir=os.path.join(BASE_PATH, '_cache')) as f:
        _, _ , net = pickle.load(f)
    return net
#----------------------------------------------------------------------------
