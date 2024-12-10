import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

# Định nghĩa ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        """Khởi tạo bộ nhớ replay với dung lượng giới hạn"""
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        """Trả về số lượng phần tử trong bộ nhớ"""
        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        """Thêm một transition vào bộ nhớ replay"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Lấy mẫu một batch từ bộ nhớ replay"""
        return random.sample(self.memory, batch_size)

    def clear(self):
        """Dọn dẹp bộ nhớ replay"""
        self.memory.clear()


# Định nghĩa mô hình DQN
class DQN(nn.Module):
    # def __init__(self, state_dim, action_dim):
    #     super(DQN, self).__init__()
    #     self.state_size = state_dim
    #     self.action_size = action_dim
        
    #      # Khởi tạo các lớp mạng
    #     self.fc1 = nn.Linear(state_dim, 128)  # state_dim là số chiều của trạng thái
    #     self.fc2 = nn.Linear(128, 128)
    #     self.fc3 = nn.Linear(128, action_dim)  # action_dim là số lượng hành động
        
    #     self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    #     self.loss_fn = nn.MSELoss()

    # def forward(self, state):
    #     x = torch.relu(self.fc1(state))
    #     x = torch.relu(self.fc2(x))
    #     return self.fc3(x)  # Output: Q-values for all actions
    
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Đảm bảo state_dim = 4
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # Output có kích thước (action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))  # Kích thước (batch_size, 128)
        x = torch.relu(self.fc2(x))      # Kích thước (batch_size, 128)
        x = self.fc3(x)                  # Kích thước (batch_size, action_dim)
        return x
    
    def select_action(state, policy_net, epsilon, action_size):
        if random.random() < epsilon:
            return torch.tensor([random.randint(0, action_size - 1)], dtype=torch.long)
        else:
            q_values = policy_net(state)  # Output của mạng có kích thước (1, action_size)
            return q_values.max(1)[1].view(1, 1)

        

# Khởi tạo DQN với các tham số state_dim và action_dim
# Ví dụ:
state_dim = 4  # Ví dụ trạng thái có 4 chiều
action_dim = 5  # Ví dụ có 5 hành động
policy_net = DQN(state_dim=state_dim, action_dim=action_dim)

# In ra cấu trúc của mô hình
print(policy_net)
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         """Xử lý thông qua các lớp fully connected"""
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


# Hàm chọn hành động
def select_action(state, policy_net, epsilon, action_size):
    # Đảm bảo rằng state có đúng kích thước
    q_values = policy_net(state)  # Đầu ra từ mạng neural (batch_size, action_size)
    print("q_values shape:", q_values.shape)  # In kích thước của q_values

    if random.random() < epsilon:
        # Chọn hành động ngẫu nhiên
        return random.randint(0, action_size - 1)
    else:
        # Chọn hành động với giá trị Q lớn nhất
        # max(1) tìm giá trị lớn nhất theo chiều 1, trả về (value, index), ta lấy index (hành động)
        if q_values.dim() > 1:  # Nếu có nhiều chiều
            return q_values.max(1)[1].item()  # Trả về index của hành động
        else:  # Nếu chỉ có 1 chiều
            return q_values.max(0)[1].item()  # Trả về index của hành động




# Hàm tối ưu hóa mô hình DQN
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    """
    Tối ưu hóa mô hình bằng cách cập nhật các tham số của policy network
    theo hàm mất mát giữa giá trị Q dự đoán và giá trị Q mục tiêu.
    """
    if len(memory) < batch_size:
        return

    # Lấy mẫu một batch từ bộ nhớ replay
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])
    actions = torch.tensor(batch[1])
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])
    dones = torch.tensor(batch[4], dtype=torch.bool)

    # Tính giá trị Q cho các trạng thái hiện tại
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Tính giá trị Q cho các trạng thái tiếp theo từ mạng mục tiêu
    next_q_values = target_net(next_states).max(1)[0].detach()  # Giá trị Q tối đa của các trạng thái tiếp theo
    next_q_values[dones] = 0.0  # Nếu episode kết thúc, giá trị Q là 0

    # Tính giá trị mục tiêu
    q_targets = rewards + gamma * next_q_values

    # Tính hàm mất mát
    loss = torch.nn.functional.mse_loss(q_values, q_targets)

    # Cập nhật các tham số của policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Hàm cập nhật mạng mục tiêu
def update_target_network(policy_net, target_net):
    """Cập nhật trọng số của mạng mục tiêu từ mạng chính (policy_net)"""
    target_net.load_state_dict(policy_net.state_dict())



# # dqn.py
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import namedtuple, deque

# # Định nghĩa ReplayMemory
# class ReplayMemory:
#     def __init__(self, capacity):
#         # Initialize the memory with a specified capacity
#         self.memory = deque(maxlen=capacity)
    
#     def __len__(self):
#         # Implement __len__ to return the number of stored transitions
#         return len(self.memory)

#     def push(self, state, action, reward, next_state, done):
#         # Store a transition in memory
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         # Sample a batch of transitions
#         return random.sample(self.memory, batch_size)

#     def clear(self):
#         # Optionally, clear the memory
#         self.memory.clear()


# # Định nghĩa DQN model
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # Hàm chọn hành động
# # def select_action(state, policy_net, epsilon):
# #     if random.random() > epsilon:
# #         with torch.no_grad():
# #             return policy_net(state).max(1)[1].view(1, 1)
# #     else:
# #         return torch.tensor([[random.randrange(policy_net.out_features)]], dtype=torch.long)
# def select_action(state, policy_net, epsilon, action_size):
#     if random.random() > epsilon:
#         # Dự đoán hành động dựa trên policy
#         with torch.no_grad():
#             state = state.unsqueeze(0)  # Nếu state là vector, chuyển thành batch
#             return policy_net(state).max(1)[1].view(1, 1)  # Chọn hành động với giá trị lớn nhất
#     else:
#         # Chọn hành động ngẫu nhiên
#         return torch.tensor([[random.choice(range(action_size))]], dtype=torch.long)




