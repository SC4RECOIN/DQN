import numpy as np
import random
import torch
from torch.nn import functional
from torch import optim
from collections import deque

from model import QNetwork, DuelingQNetwork


class Agent(object):
    def __init__(self, state_size, action_size,  mem_length=100000, ddqn=True):
        self.gamma = 0.99
        self.batch_size = 64
        self.action_size = action_size
        self.ddqn = ddqn

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if ddqn:
            self.model = DuelingQNetwork(state_size, action_size).to(self.device)
            self.target_model = DuelingQNetwork(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
            self.experience = self.ddqn_experience
        else:
            self.model = QNetwork(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
            self.experience = self.dqn_experience

        # replay memory
        self.memory = deque(maxlen=mem_length) 

    def act(self, state, eps=0):
        # epsilon greedy        
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))

        # state to predict action from
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        
        self.model.train()
        return np.argmax(action_values.cpu().data.numpy())
    
    def ddqn_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return 

        # get random batch
        states, actions, rewards, next_states, terminals = self.get_batch()

        # Get expected Q values from local model
        expected = self.model(states).gather(1, actions)  
        Q = self.model(next_states).detach()

        # Get max predicted Q values (for next states) from target model
        targets_next = self.target_model(next_states).detach()
        targets_next = targets_next.gather(1, Q.max(1)[1].unsqueeze(1))

        # Compute Q targets for current states 
        targets = rewards + (self.gamma * targets_next * (1 - terminals))
        
        # compute loss
        loss = functional.mse_loss(expected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        lr = 0.001
        for target_param, primary_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(lr * primary_param.data + (1-lr) * target_param.data) 

    def dqn_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return 

        # get random batch
        states, actions, rewards, next_states, terminals = self.get_batch()

        Q = self.model.forward(states)
        Q = Q.gather(1, actions).squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected = rewards.squeeze(1) + self.gamma * max_next_Q

        # update model
        loss = functional.mse_loss(Q, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_batch(self):
        experiences = np.array(random.sample(self.memory, k=self.batch_size))
        experiences = [np.vstack(experiences[:, i]) for i in range(5)]

        # convert data to tensors
        states = torch.FloatTensor(experiences[0]).to(self.device)
        actions = torch.LongTensor(experiences[1]).to(self.device)
        rewards = torch.FloatTensor(experiences[2]).to(self.device)
        next_states = torch.FloatTensor(experiences[3]).to(self.device)
        terminals = torch.FloatTensor(experiences[4].astype(np.uint8)).to(self.device)

        return states, actions, rewards, next_states, terminals
