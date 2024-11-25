import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, select_action
from environment import Environment

# Thông số
width, height = 800, 600
env = Environment(width, height)
state_size = len(env.reset())
action_size = 5
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
target_update = 10

# Khởi tạo DQN
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

# Hàm tối ưu hóa
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])
    actions = torch.tensor(batch[1])
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])
    dones = torch.tensor(batch[4], dtype=torch.bool)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0].detach()
    next_q_values[dones] = 0.0
    q_targets = rewards + gamma * next_q_values

    loss = torch.nn.functional.mse_loss(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Vòng lặp huấn luyện
for episode in range(100):
    state = env.reset()
    total_reward = 0

    for t in range(200):
        action = select_action(torch.tensor(state, dtype=torch.float32), policy_net, epsilon, action_size)
        action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]
        next_state, reward, done = env.step(action_mapping[action])
        memory.push(state, action, reward, next_state, done)
        state = next_state

        if len(memory) >= batch_size:
            optimize_model()

        total_reward += reward
        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total Reward: {total_reward}")

# Lưu policy_net sau khi huấn luyện
torch.save(policy_net.state_dict(), "policy_net.pth")
print("Mô hình đã được lưu vào file policy_net.pth")

