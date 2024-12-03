import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Định nghĩa lớp Actor để chọn hành động từ trạng thái
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # Lớp fully-connected đầu tiên
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Lớp fully-connected thứ hai
        self.fc3 = nn.Linear(hidden_size, action_size)  # Lớp fully-connected để đưa ra hành động
        self.softmax = nn.Softmax(dim=-1)  # Áp dụng hàm Softmax để chọn hành động từ phân phối xác suất

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # Áp dụng ReLU activation function
        x = torch.relu(self.fc2(x))  # Áp dụng ReLU activation function lần thứ hai
        x = self.fc3(x)  # Lớp cuối để tính ra giá trị của mỗi hành động
        return self.softmax(x)  # Trả về xác suất của mỗi hành động từ phân phối softmax

# Định nghĩa lớp Critic để đánh giá giá trị của trạng thái
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # Lớp fully-connected đầu tiên
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Lớp fully-connected thứ hai
        self.fc3 = nn.Linear(hidden_size, 1)  # Lớp fully-connected để đưa ra giá trị trạng thái (Q-value)

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # Áp dụng ReLU activation function
        x = torch.relu(self.fc2(x))  # Áp dụng ReLU activation function lần thứ hai
        x = self.fc3(x)  # Lớp cuối để tính ra giá trị của trạng thái
        return x  # Trả về giá trị của trạng thái (Q-value)

# Định nghĩa lớp SAC (Soft Actor-Critic)
class SAC:
    def __init__(self, state_size, action_size, gamma=0.99, tau=0.005, lr=1e-3, batch_size=64):
        # Các tham số quan trọng trong thuật toán SAC
        self.gamma = gamma  # Hệ số chiết khấu cho phần thưởng
        self.tau = tau  # Tham số dùng để cập nhật dần dần các trọng số của mạng Critic mục tiêu
        self.batch_size = batch_size  # Kích thước batch khi huấn luyện
        # Khởi tạo các mạng Neural cho Actor và Critic
        self.actor = Actor(state_size, action_size)
        self.critic1 = Critic(state_size)
        self.critic2 = Critic(state_size)
        self.target_critic1 = Critic(state_size)
        self.target_critic2 = Critic(state_size)
        # Các optimizer cho Actor và Critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr)

        # Sao chép trọng số của Critic vào các mạng Critic mục tiêu
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Replay buffer lưu trữ các trải nghiệm
        self.replay_buffer = deque(maxlen=10000)

    # Phương thức chọn hành động từ mạng Actor
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Chuyển trạng thái sang tensor
        action_probs = self.actor(state_tensor)  # Tính toán phân phối xác suất của các hành động
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy()[0])  # Chọn hành động ngẫu nhiên dựa trên phân phối xác suất
        return action

    # Phương thức lưu trữ các trải nghiệm (state, action, reward, next_state, done)
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # Phương thức tối ưu hóa các mạng (Actor và Critic)
    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:  # Kiểm tra nếu không đủ batch để huấn luyện
            return

        # Lấy một batch ngẫu nhiên từ replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Tính toán giá trị của Critic
        q1_values = self.critic1(states).squeeze(1)  # Giá trị của trạng thái từ Critic 1
        q2_values = self.critic2(states).squeeze(1)  # Giá trị của trạng thái từ Critic 2

        # Tính toán Q-values của các trạng thái tiếp theo
        next_q1_values = self.target_critic1(next_states).squeeze(1)
        next_q2_values = self.target_critic2(next_states).squeeze(1)

        # Tính toán giá trị mục tiêu của Q-value
        target_q_values = rewards + self.gamma * torch.min(next_q1_values, next_q2_values) * (1 - dones.float())

        # Tính toán loss cho Critic
        critic1_loss = torch.mean((q1_values - target_q_values) ** 2)
        critic2_loss = torch.mean((q2_values - target_q_values) ** 2)

        # Tối ưu hóa Critic 1 và Critic 2
        self.optimizer_critic1.zero_grad()
        critic1_loss.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        critic2_loss.backward()
        self.optimizer_critic2.step()

        # Tính toán loss cho Actor
        action_probs = self.actor(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        q_values = torch.min(self.critic1(states), self.critic2(states)).squeeze(1)

        actor_loss = torch.mean(action_log_probs * (q_values - 0.0))

        # Tối ưu hóa Actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Cập nhật mạng Critic mục tiêu
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, environment, num_episodes=1000):
            for episode in range(num_episodes):
                state = environment.reset()
                total_reward = 0

                for t in range(200):  # Max timesteps per episode
                    action = self.select_action(state)
                    next_state, reward, done = environment.step(action)
                    self.store_experience(state, action, reward, next_state, done)
                    self.optimize()

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                print(f"Episode {episode} - Total Reward: {total_reward}")