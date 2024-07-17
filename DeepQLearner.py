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
    def __init__(self, features = 2, actions = 3):
        super(DeepQNetwork, self).__init__()
        self.input_layer = nn.Linear(features, 8)
        self.hidden1 = nn.Linear(8, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output_layer = nn.Linear(8, actions)
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
        # self.filled = False
        self.capacity = capacity
        # self.position = 0
        self.mem_counter = 0
        self.state_memory = np.empty((capacity, features), dtype=np.float32)
        self.new_state_memory = np.empty((capacity, features), dtype=np.float32)
        self.action_memory = np.empty(capacity, dtype=np.int32)
        self.reward_memory = np.empty(capacity, dtype=np.float32)
        return
        
    def add(self, experience):
        index = self.mem_counter % self.capacity

        self.state_memory[index] = experience[0]
        self.action_memory[index] = experience[1]
        self.new_state_memory[index] = experience[2]
        self.reward_memory[index] = experience[3]
        self.mem_counter += 1
        return
        
    def sample(self, batch_size):
        # To Do: Implement
        max_mem = min(self.mem_counter, self.capacity)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        return states, actions, new_states, rewards
    



class DeepQLearner:

    def __init__(self, features = 2, actions = 3, gamma = 0.99, epsilon = 0.999,
                 epsilon_decay = 0.99999, tau = 0.1, batch_size = 1024, lr = 0.0005,
                 mem_capacity = 10000):    # Add params
    
        # Make networks, store parameters, initialize things.

        self.device = torch.device("cpu")
        self.prev_state = None
        self.prev_action = None
        self.min_epsilon = 0.05
        self.loss = -99999
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        # self.mem_counter = 0
        self.real_experiences = 0
        self.memory = ReplayMemory(mem_capacity, features)
        self.Qnet = DeepQNetwork(features=features, actions=actions)
        self.Tnet = DeepQNetwork(features=features, actions=actions)
        self.Qnet.to(self.device)
        self.Tnet.to(self.device)
        if lr != None:
            self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=lr)
        else:    
            self.optimizer = torch.optim.Adam(self.Qnet.parameters())


    # Probably will have several helper methods
    
    
    def update_target_network(self):
        for target_param, q_param in zip(self.Tnet.parameters(), self.Qnet.parameters()):
            # print(f"\nTarget param: {target_param}, Q param: {q_param}")
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
            # print(f"Target param: {target_param}, Q param: {q_param}")
        return
    
        
    def replay(self):
        
        states, actions, next_states, rewards = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)

        # Run through the Q network, let it evaluate what it currently thinks about each state
        curr_q = self.Qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            
            # Ask the Q network what it thinks is the best action for the new state
            predicted_q = self.Qnet(next_states)
            next_actions = torch.argmax(predicted_q, dim=1)
            
            # Ask the T network what it thinks about those actions
            target_q = self.Tnet(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate our "y" to compare to our curr_q (y_pred)
            expected_q = rewards + self.gamma * target_q

        # Loss and backprop
        loss = torch.nn.MSELoss()
        graph = loss(curr_q, expected_q)
        self.optimizer.zero_grad()
        graph.backward()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(), 5)
        self.optimizer.step() 
        self.loss = graph.item()
        
        return


    def train(self, s, r):    # Add params

        # Select a real action, maybe at random.
        # Remember what we've done in an experience buffer.
        # Occasionally sample and train on a bunch of stuff from the buffer.
        
        # Select an action
        self.real_experiences += 1
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        if np.random.rand() <= self.epsilon:
            a = np.random.randint(self.actions)
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
        else:    
            with torch.no_grad():
                actions = self.Qnet(s)
                a = torch.argmax(actions).item()
        
        
        # Remember the experience
        experience = [self.prev_state, self.prev_action, s, r]
        self.memory.add(experience)
        if self.real_experiences > 1999:
            self.real_experiences = 0
            self.replay()
        
        # Update T net
        self.update_target_network()

        self.prev_state = s.numpy()
        self.prev_action = a

        # Eventually...
        return a


    def test(self, s, allow_random = False):   # Plus whatever else you need.

        # Select a real action, maybe at random (sometimes).
        # No experience buffer here.  No updating the networks.
        
        if allow_random and np.random.rand() <= self.epsilon:
            a = np.random.randint(self.actions)
            self.epsilon *= self.epsilon_decay
        else: 
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                actions = self.Qnet(s)
                a = torch.argmax(actions).item()
                s = s.numpy()

        self.prev_state = s
        self.prev_action = a
        return a


    def reset_loss(self):
        self.loss = -99999
        return

