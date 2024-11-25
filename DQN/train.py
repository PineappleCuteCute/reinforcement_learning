import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, select_action
from environment import Environment
import torch.nn.functional as F

# Khởi tạo môi trường
width, height = 800, 600
env = Environment(width, height)

def optimize_model(policy_net, target_net, memory, optimizer, gamma):
    if len(memory) < batch_size:
        return

    # Lấy mẫu từ Replay Memory
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])
    actions = torch.tensor(batch[1], dtype=torch.long)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])
    dones = torch.tensor(batch[4], dtype=torch.bool)

    # Tính giá trị Q từ policy_net
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Tính giá trị Q mục tiêu từ target_net
    next_q_values = target_net(next_states).max(1)[0].detach()
    next_q_values[dones] = 0.0  # Nếu trạng thái kết thúc, không cộng thêm giá trị Q
    q_targets = rewards + gamma * next_q_values

    # Tính loss giữa Q hiện tại và Q mục tiêu
    loss = F.mse_loss(q_values, q_targets)

    # Tối ưu hóa mạng
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Thông số
width, height = 800, 600
# Lấy state_size từ kích thước state thực tế
state_size = len(env.reset())
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
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        # Lưu vào Replay Memory
        memory.push(state, action, reward, next_state, done)
        state = next_state

        # Tối ưu hóa mô hình
        optimize_model(policy_net, target_net, memory, optimizer, gamma)

        total_reward += reward
        if done:
            break

    # Giảm epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Cập nhật target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total Reward: {total_reward}")
    print(f"State Size: {state_size}, Action Size: {action_size}")
