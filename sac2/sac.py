# sac.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np  # Thêm dòng này để sử dụng np

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        # Lớp FC xử lý state
        self.state_fc = nn.Linear(state_size, 64)  
        # Lớp FC xử lý action
        self.action_fc = nn.Linear(action_size, 64)  
        # Lớp FC để tính giá trị Q từ state và action đã nối
        self.q_value_fc = nn.Linear(128, 1)  

    def forward(self, state, action):
        # Xử lý state
        state_value = torch.relu(self.state_fc(state))  
        # Xử lý action
        action_value = torch.relu(self.action_fc(action))  

        # Đảm bảo rằng action có kích thước (batch_size, action_size)
        if len(action.shape) == 1:  # Nếu action có dạng (batch_size,)
            action = action.unsqueeze(1)  # Thêm một chiều để có dạng (batch_size, action_size)
        
        # Nối state_value và action_value
        combined = torch.cat([state_value, action_value], dim=-1)  # Nối state và action
        
        # Tính toán giá trị Q
        q_value = self.q_value_fc(combined)  
        return q_value




class SAC:
    def __init__(self, state_size, action_size, buffer_size=1000000, batch_size=64):
        # Các tham số kích thước và các tham số huấn luyện
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size  # Kích thước của replay buffer
        self.batch_size = batch_size    # Kích thước batch

        # Khởi tạo replay buffer (có thể dùng deque hoặc list)
        self.replay_buffer = deque(maxlen=self.buffer_size)

        # Các thành phần khác của SAC
        self.actor = Actor(state_size, action_size)
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)

    def store_experience(self, state, action, reward, next_state, done):
        # Lưu trữ trải nghiệm vào replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        action_probs = action_probs.cpu().detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action
    
    # def store_experience(self, state, action, reward, next_state, done):
    #     self.replay_buffer.append((state, action, reward, next_state, done))
    def store_experience(self, state, action, reward, next_state, done):
        # Lưu trữ trải nghiệm vào replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def optimize(self):
        # Lấy một batch ngẫu nhiên từ replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển các giá trị này thành tensor
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Lấy giá trị Q cho critic1 và critic2 bằng cách truyền vào cả state và action
        critic1_output = self.critic1(states, actions)  # Truyền vào state và action
        critic2_output = self.critic2(states, actions)  # Truyền vào state và action

        # Tiến hành tính toán hàm mất mát (loss) và cập nhật weights cho critic
        # Ví dụ đơn giản với loss MSE
        target_q_value = rewards + (1 - dones) * self.gamma * torch.min(self.target_critic1(next_states, actions), self.target_critic2(next_states, actions))
        critic1_loss = torch.mean((critic1_output - target_q_value) ** 2)
        critic2_loss = torch.mean((critic2_output - target_q_value) ** 2)

        # Cập nhật các critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Cập nhật target critic
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic1_loss.item(), critic2_loss.item()


    
    def update_target_network(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
