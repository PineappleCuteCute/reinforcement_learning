# sac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Khởi tạo mạng Actor và Critic
        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.critic_target = self.create_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay Buffer
        self.buffer = []  # Nên sử dụng một lớp Replay Buffer đầy đủ chức năng

    def create_actor(self):
        # Mạng Actor đơn giản
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh()  # Đảm bảo hành động nằm trong phạm vi [-1, 1]
        )

    def create_critic(self):
        # Mạng Critic đơn giản
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def select_action(self, state):
        """Chọn hành động từ chính sách của SAC"""
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(state)
        return action.argmax().item()  # Trả về index của hành động có giá trị cao nhất

    def train(self):
        """Huấn luyện SAC agent (phần này cần được triển khai đầy đủ)"""
        if len(self.buffer) < 32:
            return

        # Sample từ replay buffer
        batch = np.random.choice(self.buffer, 32, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Tính Q-values cho trạng thái hiện tại
        current_q = self.critic(torch.cat([states, actions.unsqueeze(1).float()], dim=1)).squeeze()

        # Tính Q-values cho trạng thái tiếp theo từ mạng mục tiêu
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q = self.critic_target(torch.cat([next_states, next_actions], dim=1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss cho Critic
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Tối ưu hóa Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Loss cho Actor
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()

        # Tối ưu hóa Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Cập nhật mạng mục tiêu
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    def add_experience(self, state, action, reward, next_state, done):
        """Thêm trải nghiệm vào replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > 10000:
            self.buffer.pop(0)
