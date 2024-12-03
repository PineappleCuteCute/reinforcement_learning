# dqn.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

# Định nghĩa ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        # Initialize the memory with a specified capacity
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        # Implement __len__ to return the number of stored transitions
        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        # Store a transition in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Sample a batch of transitions
        return random.sample(self.memory, batch_size)

    def clear(self):
        # Optionally, clear the memory
        self.memory.clear()


# Định nghĩa DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hàm chọn hành động
# def select_action(state, policy_net, epsilon):
#     if random.random() > epsilon:
#         with torch.no_grad():
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(policy_net.out_features)]], dtype=torch.long)
def select_action(state, policy_net, epsilon, action_size):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)  # Lấy hành động có giá trị lớn nhất
    else:
        return torch.tensor([[random.randrange(action_size)]], dtype=torch.long)  # Chọn ngẫu nhiên hành động



