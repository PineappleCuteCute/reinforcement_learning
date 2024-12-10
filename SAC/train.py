import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, select_action, optimize_model, update_target_network
# Đảm bảo rằng bạn import đúng các lớp cần thiết
from sac_agent import SACAgent  # SAC agent
from environment import Environment  # Environment
from robot import Robot
# Định nghĩa số chiều của trạng thái và hành động
state_dim = 2  # Tọa độ x, y của robot
action_dim = 2  # Hành động di chuyển trong 2 chiều: [dx, dy]

# Khởi tạo SAC agent
sac_agent = SACAgent(state_dim, action_dim)

# Khởi tạo môi trường
width = 800
height = 600
env = Environment(width, height)  # Không truyền sac_agent vào constructor của Environment

# Khởi tạo robot và truyền sac_agent vào
env.robot = Robot(env.cols // 2 * env.cell_size, env.rows // 2 * env.cell_size, size=10, sac_agent=sac_agent)

# Thông số
width, height = 800, 600  # Kích thước màn hình
# env = Environment(width, height)  # Khởi tạo môi trường
state_size = len(env.reset())  # Kích thước của trạng thái, tương ứng với các thông tin về môi trường
action_size = 5  # Tổng số hành động có thể: [di chuyển lên, xuống, trái, phải, đứng im]
gamma = 0.99  # Hệ số chiết khấu
epsilon = 1.0  # Tỷ lệ epsilon cho epsilon-greedy (ban đầu cao để khám phá)
epsilon_decay = 0.995  # Tốc độ giảm epsilon
epsilon_min = 0.1  # Epsilon tối thiểu
batch_size = 64  # Kích thước của batch khi huấn luyện
target_update = 10  # Cập nhật mạng mục tiêu sau mỗi số bước nhất định

# Khởi tạo mô hình DQN và bộ nhớ Replay
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # Đồng bộ hóa mạng mục tiêu
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)  # Tối ưu hóa với Adam
memory = ReplayMemory(10000)  # Bộ nhớ với dung lượng 10,000 trải nghiệm

# Hàm tối ưu hóa mô hình
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)  # Lấy mẫu từ bộ nhớ
    batch = list(zip(*transitions))  # Tách các phần tử trong batch
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

# Vòng lặp huấn luyện
for episode in range(1000):
    state = env.reset()  # Khởi tạo trạng thái ban đầu
    total_reward = 0

    for t in range(200):  # Mỗi episode có tối đa 200 bước
        # Chọn hành động theo chiến lược epsilon-greedy
        action = select_action(torch.tensor(state, dtype=torch.float32), policy_net, epsilon, action_size)

        # Chuyển đổi hành động (mảng di chuyển) và thực hiện bước trong môi trường
        action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # Các hành động di chuyển [up, down, left, right, stay]
        next_state, reward, done = env.step(action_mapping[action])  # Không cần gọi item() nữa
 # Thực hiện hành động và nhận trạng thái tiếp theo

        # Lưu trữ transition vào bộ nhớ replay
        memory.push(state, action, reward, next_state, done)  # Không cần gọi item() nữa


        state = next_state  # Cập nhật trạng thái

        # Tối ưu hóa mô hình nếu bộ nhớ đủ lớn
        if len(memory) >= batch_size:
            optimize_model()

        total_reward += reward  # Cộng dồn phần thưởng

        if done:  # Nếu episode kết thúc, dừng vòng lặp
            break

    # Giảm dần epsilon để giảm tần suất hành động ngẫu nhiên
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Cập nhật mạng mục tiêu mỗi vài episode
    if episode % target_update == 0:
        update_target_network(policy_net, target_net)

    print(f"Episode {episode}: Total Reward: {total_reward}")

# Lưu mô hình sau khi huấn luyện
torch.save(policy_net.state_dict(), "policy_net.pth")
print("Mô hình đã được lưu vào file policy_net.pth")



# import torch
# import torch.optim as optim
# from dqn import DQN, ReplayMemory, select_action
# from environment import Environment

# # Thông số
# width, height = 800, 600
# env = Environment(width, height)
# state_size = len(env.reset())
# print(f"State Size During Training: {state_size}")
# action_size = 5  # Tổng số hành động (di chuyển lên, xuống, trái, phải, đứng im)
# gamma = 0.99  # Hệ số chiết khấu
# epsilon = 1.0  # Tỷ lệ ngẫu nhiên (epsilon-greedy)
# epsilon_decay = 0.995  # Tốc độ giảm epsilon
# epsilon_min = 0.1  # Epsilon tối thiểu
# batch_size = 64  # Kích thước batch cho huấn luyện
# target_update = 10  # Số lần update mạng mục tiêu

# # Khởi tạo DQN
# policy_net = DQN(state_size, action_size)
# target_net = DQN(state_size, action_size)
# target_net.load_state_dict(policy_net.state_dict())  # Đồng bộ mạng mục tiêu
# optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
# memory = ReplayMemory(10000)  # Bộ nhớ để lưu các transition

# # Hàm tối ưu hóa
# def optimize_model():
#     if len(memory) < batch_size:
#         return

#     transitions = memory.sample(batch_size)  # Sample một batch
#     batch = list(zip(*transitions))  # Tách các phần tử trong batch
#     states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])  # Chuyển đổi trạng thái thành tensor
#     actions = torch.tensor(batch[1])  # Chuyển đổi hành động thành tensor
#     rewards = torch.tensor(batch[2], dtype=torch.float32)  # Phần thưởng
#     next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])  # Trạng thái kế tiếp
#     dones = torch.tensor(batch[4], dtype=torch.bool)  # Tính kết thúc episode

#     # Tính giá trị Q cho các trạng thái hiện tại
#     q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
#     # Tính giá trị Q cho các trạng thái tiếp theo (từ mạng mục tiêu)
#     next_q_values = target_net(next_states).max(1)[0].detach()  # Tách các giá trị max từ các hành động
#     next_q_values[dones] = 0.0  # Nếu episode kết thúc, không tính giá trị Q cho các hành động tiếp theo

#     q_targets = rewards + gamma * next_q_values  # Công thức tính giá trị mục tiêu

#     # Tính hàm mất mát
#     loss = torch.nn.functional.mse_loss(q_values, q_targets)
    
#     # Cập nhật các tham số của policy_net
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # Vòng lặp huấn luyện
# for episode in range(100):
#     state = env.reset()
#     total_reward = 0

#     for t in range(200):  # Số bước trong mỗi episode
#         # Chọn hành động bằng cách sử dụng epsilon-greedy policy
#         action = select_action(torch.tensor(state, dtype=torch.float32), policy_net, epsilon, action_size)



        
#         # Mảng các hành động di chuyển (có thể mở rộng thêm các hành động)
#         action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
#         next_state, reward, done = env.step(action_mapping[action])  # Thực hiện hành động và nhận trạng thái tiếp theo
#         memory.push(state, action, reward, next_state, done)  # Lưu trữ transition vào bộ nhớ
        
#         state = next_state

#         # Tối ưu hóa mô hình khi bộ nhớ đủ lớn
#         if len(memory) >= batch_size:
#             optimize_model()

#         total_reward += reward  # Cộng dồn phần thưởng

#         if done:  # Nếu kết thúc episode
#             break

#     # Giảm dần epsilon
#     epsilon = max(epsilon * epsilon_decay, epsilon_min)

#     # Cập nhật mạng mục tiêu sau mỗi vài episode
#     if episode % target_update == 0:
#         target_net.load_state_dict(policy_net.state_dict())

#     print(f"Episode {episode}: Total Reward: {total_reward}")

# # Lưu mô hình sau khi huấn luyện
# torch.save(policy_net.state_dict(), "policy_net.pth")
# print("Mô hình đã được lưu vào file policy_net.pth")
