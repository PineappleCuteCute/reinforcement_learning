import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, select_action
from environment import Environment

# Thông số
width, height = 800, 600
state_size = 8  # Định nghĩa kích thước state
action_size = 5  # Các hành động: lên, xuống, trái, phải, giữ nguyên
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
target_update = 10

# Tạo môi trường
env = Environment(width, height)

# Khởi tạo DQN
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

# Huấn luyện
for episode in range(1000):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Giới hạn mỗi episode 200 bước
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(state_tensor, policy_net, epsilon, action_size)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Lưu vào Replay Memory
        memory.push(state, action, reward, next_state, done)
        state = next_state

        # Tối ưu hóa DQN
        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            optimize_model(policy_net, target_net, transitions, optimizer, gamma)

        if done:
            break

    # Giảm epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Cập nhật target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")
