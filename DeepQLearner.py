#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:13:55 2024

@author: pmullin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):

    # Need init method to define the layers.

    # Need forward method to do the forward computation.

    pass


class DeepQLearner:

    def __init__(self):    # Add params
        self.device = torch.device("cpu")

        # Make networks, store parameters, initialize things.


    # Probably will have several helper methods


    def train(self, s, r):    # Add params

        # Select a real action, maybe at random.
        # Remember what we've done in an experience buffer.
        # Occasionally sample and train on a bunch of stuff from the buffer.

        # Eventually...
        return a


    def test(self, s, allow_random = False):   # Plus whatever else you need.

        # Select a real action, maybe at random (sometimes).
        # No experience buffer here.  No updating the networks.

        # Eventually...
        return a

