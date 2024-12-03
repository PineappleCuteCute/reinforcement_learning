import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, select_action
from environment import Environment

# Thông số
width, height = 800, 600
env = Environment(width, height)
state_size = len(env.reset())
print(f"State Size During Training: {state_size}")
action_size = 5  # Tổng số hành động (di chuyển lên, xuống, trái, phải, đứng im)
gamma = 0.99  # Hệ số chiết khấu
epsilon = 1.0  # Tỷ lệ ngẫu nhiên (epsilon-greedy)
epsilon_decay = 0.995  # Tốc độ giảm epsilon
epsilon_min = 0.1  # Epsilon tối thiểu
batch_size = 64  # Kích thước batch cho huấn luyện
target_update = 10  # Số lần update mạng mục tiêu

# Khởi tạo DQN
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # Đồng bộ mạng mục tiêu
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)  # Bộ nhớ để lưu các transition

# Hàm tối ưu hóa
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)  # Sample một batch
    batch = list(zip(*transitions))  # Tách các phần tử trong batch
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])  # Chuyển đổi trạng thái thành tensor
    actions = torch.tensor(batch[1])  # Chuyển đổi hành động thành tensor
    rewards = torch.tensor(batch[2], dtype=torch.float32)  # Phần thưởng
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])  # Trạng thái kế tiếp
    dones = torch.tensor(batch[4], dtype=torch.bool)  # Tính kết thúc episode

    # Tính giá trị Q cho các trạng thái hiện tại
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Tính giá trị Q cho các trạng thái tiếp theo (từ mạng mục tiêu)
    next_q_values = target_net(next_states).max(1)[0].detach()  # Tách các giá trị max từ các hành động
    next_q_values[dones] = 0.0  # Nếu episode kết thúc, không tính giá trị Q cho các hành động tiếp theo

    q_targets = rewards + gamma * next_q_values  # Công thức tính giá trị mục tiêu

    # Tính hàm mất mát
    loss = torch.nn.functional.mse_loss(q_values, q_targets)
    
    # Cập nhật các tham số của policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Vòng lặp huấn luyện
for episode in range(100):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Số bước trong mỗi episode
        # Chọn hành động bằng cách sử dụng epsilon-greedy policy
        action = select_action(torch.tensor(state, dtype=torch.float32), policy_net, epsilon, action_size)



        
        # Mảng các hành động di chuyển (có thể mở rộng thêm các hành động)
        action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
        next_state, reward, done = env.step(action_mapping[action])  # Thực hiện hành động và nhận trạng thái tiếp theo
        memory.push(state, action, reward, next_state, done)  # Lưu trữ transition vào bộ nhớ
        
        state = next_state

        # Tối ưu hóa mô hình khi bộ nhớ đủ lớn
        if len(memory) >= batch_size:
            optimize_model()

        total_reward += reward  # Cộng dồn phần thưởng

        if done:  # Nếu kết thúc episode
            break

    # Giảm dần epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Cập nhật mạng mục tiêu sau mỗi vài episode
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total Reward: {total_reward}")

# Lưu mô hình sau khi huấn luyện
torch.save(policy_net.state_dict(), "policy_net.pth")
print("Mô hình đã được lưu vào file policy_net.pth")
