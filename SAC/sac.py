# sac.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Mạng Actor (chọn hành động)
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Dự đoán phân phối xác suất hành động

# Mạng Critic (đánh giá giá trị của hành động)
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Trả về giá trị Q

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Định nghĩa lớp SAC
class SAC:
    def __init__(self, state_size, action_size, gamma=0.99, tau=0.005, lr=1e-3, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Hệ số chiết khấu
        self.tau = tau  # Hệ số cập nhật mềm
        self.batch_size = batch_size  # Kích thước batch

        # Khởi tạo các mạng
        self.actor = Actor(state_size, action_size)
        self.critic1 = Critic(state_size)
        self.critic2 = Critic(state_size)
        self.target_critic1 = Critic(state_size)
        self.target_critic2 = Critic(state_size)

        # Mạng mục tiêu sao chép từ các mạng critic
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizer cho Actor và Critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr)

        # Bộ nhớ replay
        self.replay_buffer = deque(maxlen=10000)

    def select_action(self, state):
        """Chọn hành động theo chính sách epsilon-greedy từ mạng Actor"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Chuyển trạng thái thành tensor và thêm chiều batch
        action_probs = self.actor(state_tensor)  # Tính xác suất của các hành động
        action_probs = action_probs.cpu().detach().numpy()[0]  # Chuyển từ tensor sang numpy array

        # Chọn hành động dựa trên phân phối xác suất
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Lưu trữ trải nghiệm vào bộ nhớ replay"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def optimize(self):
        """Tối ưu hóa SAC bằng cách huấn luyện các mạng Actor và Critic"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Lấy mẫu batch từ bộ nhớ replay
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Tính giá trị Q cho trạng thái hiện tại từ Critic1
        q_values = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Tính giá trị Q cho trạng thái tiếp theo từ Critic1 và Critic2 (với mạng mục tiêu)
        next_q_values_1 = self.target_critic1(next_states).max(1)[0].detach()
        next_q_values_2 = self.target_critic2(next_states).max(1)[0].detach()
        next_q_values = torch.min(next_q_values_1, next_q_values_2)
        next_q_values[dones] = 0.0  # Nếu episode kết thúc, không tính giá trị Q

        # Tính giá trị mục tiêu (target Q)
        q_targets = rewards + self.gamma * next_q_values

        # Tính loss và cập nhật Critic
        critic_loss_1 = torch.nn.functional.mse_loss(q_values, q_targets)
        self.optimizer_critic1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic1.step()

        # Tính loss và cập nhật Critic2
        critic_loss_2 = torch.nn.functional.mse_loss(q_values, q_targets)
        self.optimizer_critic2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic2.step()

        # Tối ưu hóa Actor
        actor_loss = -self.critic1(states).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Cập nhật mạng mục tiêu
        self.update_target_network(self.target_critic1, self.critic1)
        self.update_target_network(self.target_critic2, self.critic2)

    def update_target_network(self, target_net, net):
        """Cập nhật mạng mục tiêu bằng cách kết hợp mạng chính và mạng mục tiêu"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque

# # Mô hình Q-Net và Policy-Net cho SAC
# class QNet(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(QNet, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)
    
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=-1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# class PolicyNet(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, action_dim)
    
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         return torch.tanh(self.fc3(x))  # Tanh để đưa giá trị hành động vào [-1, 1]

# class SAC:
#     def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4):
#         self.gamma = gamma
#         self.policy_net = PolicyNet(state_dim, action_dim)
#         self.q1_net = QNet(state_dim, action_dim)
#         self.q2_net = QNet(state_dim, action_dim)
#         self.target_q1_net = QNet(state_dim, action_dim)
#         self.target_q2_net = QNet(state_dim, action_dim)
#         self.target_q1_net.load_state_dict(self.q1_net.state_dict())
#         self.target_q2_net.load_state_dict(self.q2_net.state_dict())

#         self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
#         self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)

#     def update(self, state, action, reward, next_state, done):
#         # Tính toán giá trị mục tiêu (TD Target)
#         with torch.no_grad():
#             next_action = self.policy_net(next_state)
#             target_q1 = self.target_q1_net(next_state, next_action)
#             target_q2 = self.target_q2_net(next_state, next_action)
#             target_q = reward + self.gamma * torch.min(target_q1, target_q2) * (1 - done)
        
#         # Tính Q values cho hiện tại
#         q1_value = self.q1_net(state, action)
#         q2_value = self.q2_net(state, action)
        
#         # Tính loss và cập nhật các giá trị
#         q1_loss = nn.MSELoss()(q1_value, target_q)
#         q2_loss = nn.MSELoss()(q2_value, target_q)
        
#         # Cập nhật Q-nets
#         self.q1_optimizer.zero_grad()
#         q1_loss.backward()
#         self.q1_optimizer.step()

#         self.q2_optimizer.zero_grad()
#         q2_loss.backward()
#         self.q2_optimizer.step()

#         # Cập nhật policy network
#         policy_loss = -self.q1_net(state, self.policy_net(state)).mean()
#         self.policy_optimizer.zero_grad()
#         policy_loss.backward()
#         self.policy_optimizer.step()

#         # Cập nhật target Q-nets
#         for target, net in zip([self.target_q1_net, self.target_q2_net], [self.q1_net, self.q2_net]):
#             for target_param, param in zip(target.parameters(), net.parameters()):
#                 target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)


# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import numpy as np
# # import random
# # from collections import deque

# # # Định nghĩa lớp Actor để chọn hành động từ trạng thái
# # class Actor(nn.Module):
# #     def __init__(self, state_size, action_size, hidden_size=64):
# #         super(Actor, self).__init__()
# #         self.fc1 = nn.Linear(state_size, hidden_size)  # Lớp fully-connected đầu tiên
# #         self.fc2 = nn.Linear(hidden_size, hidden_size)  # Lớp fully-connected thứ hai
# #         self.fc3 = nn.Linear(hidden_size, action_size)  # Lớp fully-connected để đưa ra hành động
# #         self.softmax = nn.Softmax(dim=-1)  # Áp dụng hàm Softmax để chọn hành động từ phân phối xác suất

# #     def forward(self, state):
# #         x = torch.relu(self.fc1(state))  # Áp dụng ReLU activation function
# #         x = torch.relu(self.fc2(x))  # Áp dụng ReLU activation function lần thứ hai
# #         x = self.fc3(x)  # Lớp cuối để tính ra giá trị của mỗi hành động
# #         return self.softmax(x)  # Trả về xác suất của mỗi hành động từ phân phối softmax

# # # Định nghĩa lớp Critic để đánh giá giá trị của trạng thái
# # class Critic(nn.Module):
# #     def __init__(self, state_size, hidden_size=64):
# #         super(Critic, self).__init__()
# #         self.fc1 = nn.Linear(state_size, hidden_size)  # Lớp fully-connected đầu tiên
# #         self.fc2 = nn.Linear(hidden_size, hidden_size)  # Lớp fully-connected thứ hai
# #         self.fc3 = nn.Linear(hidden_size, 1)  # Lớp fully-connected để đưa ra giá trị trạng thái (Q-value)

# #     def forward(self, state):
# #         x = torch.relu(self.fc1(state))  # Áp dụng ReLU activation function
# #         x = torch.relu(self.fc2(x))  # Áp dụng ReLU activation function lần thứ hai
# #         x = self.fc3(x)  # Lớp cuối để tính ra giá trị của trạng thái
# #         return x  # Trả về giá trị của trạng thái (Q-value)

# # # Định nghĩa lớp SAC (Soft Actor-Critic)
# # class SAC:
# #     def __init__(self, state_size, action_size, gamma=0.99, tau=0.005, lr=1e-3, batch_size=64):
# #         # Các tham số quan trọng trong thuật toán SAC
# #         self.gamma = gamma  # Hệ số chiết khấu cho phần thưởng
# #         self.tau = tau  # Tham số dùng để cập nhật dần dần các trọng số của mạng Critic mục tiêu
# #         self.batch_size = batch_size  # Kích thước batch khi huấn luyện
# #         # Khởi tạo các mạng Neural cho Actor và Critic
# #         self.actor = Actor(state_size, action_size)
# #         self.critic1 = Critic(state_size)
# #         self.critic2 = Critic(state_size)
# #         self.target_critic1 = Critic(state_size)
# #         self.target_critic2 = Critic(state_size)
# #         # Các optimizer cho Actor và Critic
# #         self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
# #         self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr)
# #         self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr)

# #         # Sao chép trọng số của Critic vào các mạng Critic mục tiêu
# #         self.target_critic1.load_state_dict(self.critic1.state_dict())
# #         self.target_critic2.load_state_dict(self.critic2.state_dict())

# #         # Replay buffer lưu trữ các trải nghiệm
# #         self.replay_buffer = deque(maxlen=10000)

# #     # Phương thức chọn hành động từ mạng Actor
# #     def select_action(self, state):
# #         # Chuyển state thành tensor trước khi đưa vào mạng Actor
# #         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
# #         # Tính toán xác suất hành động từ mạng Actor
# #         action_probs = self.actor(state_tensor)  # Đầu ra từ mạng Actor

# #         # Kiểm tra kích thước của action_probs
# #         print(f"Action Probs shape: {action_probs.shape}")

# #         # Chuyển action_probs từ tensor sang mảng numpy
# #         action_probs = action_probs.cpu().detach().numpy()[0]
# #         print(f"Action Probs as numpy: {action_probs}")

# #         # Chọn hành động ngẫu nhiên dựa trên phân phối xác suất
# #         action = np.random.choice(len(action_probs), p=action_probs)

# #         return action



# #     # Phương thức lưu trữ các trải nghiệm (state, action, reward, next_state, done)
# #     def store_experience(self, state, action, reward, next_state, done):
# #         self.replay_buffer.append((state, action, reward, next_state, done))

# #     # Phương thức tối ưu hóa các mạng (Actor và Critic)
# #     def optimize(self):
# #         if len(self.replay_buffer) < self.batch_size:  # Kiểm tra nếu không đủ batch để huấn luyện
# #             return

# #         # Lấy một batch ngẫu nhiên từ replay buffer
# #         batch = random.sample(self.replay_buffer, self.batch_size)
# #         states, actions, rewards, next_states, dones = zip(*batch)

# #         states = torch.tensor(states, dtype=torch.float32)
# #         actions = torch.tensor(actions, dtype=torch.long)
# #         rewards = torch.tensor(rewards, dtype=torch.float32)
# #         next_states = torch.tensor(next_states, dtype=torch.float32)
# #         dones = torch.tensor(dones, dtype=torch.bool)

# #         # Tính toán giá trị của Critic
# #         q1_values = self.critic1(states).squeeze(1)  # Giá trị của trạng thái từ Critic 1
# #         q2_values = self.critic2(states).squeeze(1)  # Giá trị của trạng thái từ Critic 2

# #         # Tính toán Q-values của các trạng thái tiếp theo
# #         next_q1_values = self.target_critic1(next_states).squeeze(1)
# #         next_q2_values = self.target_critic2(next_states).squeeze(1)

# #         # Tính toán giá trị mục tiêu của Q-value
# #         target_q_values = rewards + self.gamma * torch.min(next_q1_values, next_q2_values) * (1 - dones.float())

# #         # Tính toán loss cho Critic
# #         critic1_loss = torch.mean((q1_values - target_q_values) ** 2)
# #         critic2_loss = torch.mean((q2_values - target_q_values) ** 2)

# #         # Tối ưu hóa Critic 1 và Critic 2
# #         self.optimizer_critic1.zero_grad()
# #         critic1_loss.backward()
# #         self.optimizer_critic1.step()

# #         self.optimizer_critic2.zero_grad()
# #         critic2_loss.backward()
# #         self.optimizer_critic2.step()

# #         # Tính toán loss cho Actor
# #         action_probs = self.actor(states)
# #         action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
# #         q_values = torch.min(self.critic1(states), self.critic2(states)).squeeze(1)

# #         actor_loss = torch.mean(action_log_probs * (q_values - 0.0))

# #         # Tối ưu hóa Actor
# #         self.optimizer_actor.zero_grad()
# #         actor_loss.backward()
# #         self.optimizer_actor.step()

# #         # Cập nhật mạng Critic mục tiêu
# #         for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
# #             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# #         for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
# #             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# #     def train(self, environment, num_episodes=1000):
# #             for episode in range(num_episodes):
# #                 state = environment.reset()
# #                 total_reward = 0

# #                 for t in range(200):  # Max timesteps per episode
# #                     action = self.select_action(state)
# #                     next_state, reward, done = environment.step(action)
# #                     self.store_experience(state, action, reward, next_state, done)
# #                     self.optimize()

# #                     state = next_state
# #                     total_reward += reward

# #                     if done:
# #                         break

# #                 print(f"Episode {episode} - Total Reward: {total_reward}")



# # # sac.py
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import numpy as np
# # import random
# # from collections import deque

# # # Định nghĩa mô hình Actor và Critic
# # class Actor(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(Actor, self).__init__()
# #         self.fc1 = nn.Linear(input_dim, 128)
# #         self.fc2 = nn.Linear(128, 64)
# #         self.fc3 = nn.Linear(64, output_dim)

# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         return torch.softmax(self.fc3(x), dim=-1)  # Dự đoán phân phối xác suất

# # class Critic(nn.Module):
# #     def __init__(self, input_dim):
# #         super(Critic, self).__init__()
# #         self.fc1 = nn.Linear(input_dim, 128)
# #         self.fc2 = nn.Linear(128, 64)
# #         self.fc3 = nn.Linear(64, 1)

# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         return self.fc3(x)

# # # Lớp SAC
# # class SAC:
# #     def __init__(self, state_dim, action_dim, alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4):
# #         self.actor = Actor(state_dim, action_dim)
# #         self.critic1 = Critic(state_dim + action_dim)
# #         self.critic2 = Critic(state_dim + action_dim)
# #         self.target_critic1 = Critic(state_dim + action_dim)
# #         self.target_critic2 = Critic(state_dim + action_dim)
# #         self.alpha = alpha  # Entropy regularization
# #         self.gamma = gamma
# #         self.tau = tau  # Mức độ cập nhật target network
# #         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
# #         self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
# #         self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

# #         # Đồng bộ target critic
# #         self.target_critic1.load_state_dict(self.critic1.state_dict())
# #         self.target_critic2.load_state_dict(self.critic2.state_dict())

# #     def update(self, state, action, reward, next_state, done):
# #         """Cập nhật các mô hình actor và critic."""
# #         # Tính Q-values cho hành động hiện tại
# #         state_action = torch.cat([state, action], dim=-1)
# #         q1 = self.critic1(state_action)
# #         q2 = self.critic2(state_action)

# #         # Tính giá trị của action tiếp theo (target)
# #         next_action_probs = self.actor(next_state)
# #         next_action = torch.multinomial(next_action_probs, 1).unsqueeze(0)

# #         # Cập nhật các giá trị mục tiêu (critic)
# #         with torch.no_grad():
# #             next_q1 = self.target_critic1(torch.cat([next_state, next_action], dim=-1))
# #             next_q2 = self.target_critic2(torch.cat([next_state, next_action], dim=-1))
# #             target_q = reward + self.gamma * (1 - done) * torch.min(next_q1, next_q2)

# #         # Tính hàm mất mát cho critic
# #         critic_loss1 = torch.mean((q1 - target_q) ** 2)
# #         critic_loss2 = torch.mean((q2 - target_q) ** 2)

# #         # Cập nhật critic
# #         self.critic1_optimizer.zero_grad()
# #         critic_loss1.backward()
# #         self.critic1_optimizer.step()

# #         self.critic2_optimizer.zero_grad()
# #         critic_loss2.backward()
# #         self.critic2_optimizer.step()

# #         # Cập nhật actor
# #         action_probs = self.actor(state)
# #         log_action_probs = torch.log(action_probs)
# #         entropy = -torch.sum(action_probs * log_action_probs)
# #         actor_loss = torch.mean(entropy - q1)

# #         self.actor_optimizer.zero_grad()
# #         actor_loss.backward()
# #         self.actor_optimizer.step()

# #         # Cập nhật target critic
# #         for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
# #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
# #         for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
# #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
