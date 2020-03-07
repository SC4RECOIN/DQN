import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return self.out(x)
        
    
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # calculates V(s)
        self.fc3_v = nn.Linear(64, 32)
        self.fc4_v = nn.Linear(32, 1)

        # calculates A(s,a)
        self.fc3_a = nn.Linear(64, 32)
        self.fc4_a = nn.Linear(32, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        v = F.relu(self.fc3_v(x))
        v = self.fc4_v(v)
        
        a = F.relu(self.fc3_a(x))
        a = self.fc4_a(a)

        return v + a - a.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
