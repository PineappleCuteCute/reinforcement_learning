import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym  # Thêm thư viện gym cho môi trường

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
        # Kiểm tra state có phải là mảng numpy hoặc danh sách hợp lệ
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        elif isinstance(state, list):
            state = torch.tensor(np.array(state), dtype=torch.float32)
        else:
            raise ValueError(f"Expected state to be a list or numpy array, but got {type(state)}")

        # In ra state
        print("State:", state)

        # Dự đoán hành động từ actor
        with torch.no_grad():
            action = self.actor(state)

        # In ra giá trị của action (hành động)
        print("Action values:", action)

        # Trả về chỉ số của hành động có giá trị cao nhất
        return action.argmax().item()


# Tạo môi trường CartPole-v1
env = gym.make('CartPole-v1')

# Khởi tạo agent
state_dim = env.observation_space.shape[0]  # Kích thước state (CartPole có 4 đặc trưng)
action_dim = env.action_space.n  # Lấy số hành động có thể thực hiện từ môi trường
agent = SACAgent(state_dim, action_dim)

# Lấy state từ môi trường (lấy cả state và info)
state, info = env.reset()

# Kiểm tra lại state (nó phải là một numpy array hoặc list)
print("State type:", type(state))

# Gọi hàm select_action và in ra kết quả
action_index = agent.select_action(state)
print("Selected action index:", action_index)

# Tiến hành bước tiếp theo trong môi trường (bước mô phỏng)
next_state, reward, done, info = env.step(action_index)

# In ra kết quả bước tiếp theo
print("Next state:", next_state)
print("Reward:", reward)
