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
import numpy as np

class DeepQNetwork(nn.Module):

    # Need init method to define the layers.
    def __init__(self, features = 6, actions = 3):
        super(DeepQNetwork, self).__init__()
        self.input_layer = nn.Linear(features, 10)
        self.hidden1 = nn.Linear(10, 10)
        self.hidden2 = nn.Linear(10, 8)
        self.hidden3 = nn.Linear(8, 6)
        self.output_layer = nn.Linear(6, actions)
        return

    # Need forward method to do the forward computation.
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output_layer(x)
        
        
class ReplayMemory:
    
    def __init__(self, capacity, features):
        self.filled = False
        self.capacity = capacity
        self.position = 0
        self.state_memory = np.empty((capacity, features), dtype=np.float32)
        self.new_state_memory = np.empty((capacity, features), dtype=np.float32)
        self.action_memory = np.empty(capacity, dtype=np.int32)
        self.reward_memory = np.empty(capacity, dtype=np.float32)
        return
        
    def add(self, experience):
        if self.position == self.capacity:
            self.position = 0
            self.filled = True
        self.state_memory[self.position] = experience[0]
        self.new_state_memory[self.position] = experience[2]
        self.action_memory[self.position] = experience[1]
        self.reward_memory[self.position] = experience[3]
        return
        
    def sample(self, batch_size):
        # To Do: Implement
        max_mem = min(self.position, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        return states, actions, new_states, rewards
    



class DeepQLearner:

    def __init__(self, features = 6, actions = 3, gamma = 0.99, epsilon = 0.98,
                 epsilon_decay = 0.999, tau = 0.005, batch_size = 128, lr = None,
                 momentum = None, mem_capacity = 10000):    # Add params
    
        # Make networks, store parameters, initialize things.

        self.device = torch.device("cpu")
        self.prev_state = None
        self.prev_action = None
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.real_experiences = 0
        self.memory = ReplayMemory(mem_capacity, features)
        self.Qnet = DeepQNetwork(features=features, actions=actions)
        self.Tnet = DeepQNetwork(features=features, actions=actions)
        self.Qnet.to(self.device)
        self.Tnet.to(self.device)
        if lr != None and momentum != None:
            self.optimizer = torch.optim.SGD(self.Qnet, lr=lr, momentum=momentum)
        else:    
            self.optimizer = torch.optim.SGD(self.Qnet)

        
        


    # Probably will have several helper methods
    
    
    def update_target_network(self):
        for target_param, q_param in zip(self.Tnet.parameters(), self.Qnet.parameters()):
            target_param.data.copy(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        return
    
    def select_action(self, s, allow_random = True):
        
        if allow_random and np.random.rand() <= self.epsilon:
            a = np.random.randint(self.actions)
            self.epsilon *= self.epsilon_decay
        else:    
            s = torch.tensor(s).to(self.device)
            actions = self.Qnet(s)
            a = torch.argmax(actions).numpy()
        
        return a
            
    def replay(self):
        if not self.memory.filled:
            return
        
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)

        y_pred = self.Qnet(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        y = self.Tnet(next_states).max(1)[0]
        y_expected = rewards + self.gamma * y

        loss = torch.nn.MSELoss()
        graph = loss(y_pred, y_expected)
        graph.backward()
        self.optimizer.step()        


    def train(self, s, r):    # Add params

        # Select a real action, maybe at random.
        # Remember what we've done in an experience buffer.
        # Occasionally sample and train on a bunch of stuff from the buffer.
        
        # Select an action
        self.real_experiences += 1
        y_pred = self.select_action(self.prev_state)
        y = torch.argmax(self.Tnet(s)).numpy()
        y_expected = r + self.gamma * y
        
        # Remember the experience
        experience = [self.prev_state, self.prev_action, s, r]
        self.memory.add(experience)
        if self.real_experiences > 1000:
            self.real_experiences = 0
            self.replay()
            
        # Calculate loss from "correct y value" gotten from target network
        loss = torch.nn.MSELoss()
        graph = loss(y_pred, y)
        graph.backward()
        self.optimizer.step()
        
        # Update T net and set out prev state and action
        self.update_target_network()
        self.prev_state = s
        self.prev_action = y_pred

        # Eventually...
        return y_pred


    def test(self, s, allow_random = False):   # Plus whatever else you need.

        # Select a real action, maybe at random (sometimes).
        # No experience buffer here.  No updating the networks.
        
        a = self.select_action(s, allow_random=allow_random)
        self.optimizer.zero_grad()
        self.prev_state = s
        self.prev_action = a
        return a


