import pygame
import torch
from environment import Environment
from dqn import DQN, ReplayMemory, select_action
import numpy as np

# Thông số màn hình
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CELL_SIZE = 20

# Tạo môi trường
env = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)

# Thông số DQN
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
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

# Khởi tạo Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('DQN Simulation')
clock = pygame.time.Clock()

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Vẽ môi trường
def draw_environment():
    screen.fill(WHITE)

    # Vẽ chướng ngại vật tĩnh
    for obs in env.static_obstacles:
        pygame.draw.rect(screen, BLUE, (obs['position'][0], obs['position'][1], CELL_SIZE, CELL_SIZE))

    # Vẽ chướng ngại vật động
    for obs in env.dynamic_obstacles:
        pygame.draw.rect(screen, RED, (obs['position'][0], obs['position'][1], CELL_SIZE - 5, CELL_SIZE - 5))

    # Vẽ robot
    robot_pos = env.robot.get_position()
    pygame.draw.circle(screen, GREEN, (int(robot_pos[0]), int(robot_pos[1])), env.robot.size)

    # Vẽ mục tiêu
    pygame.draw.rect(screen, YELLOW, (env.goal[0], env.goal[1], CELL_SIZE, CELL_SIZE))

# Hàm tối ưu hóa mô hình
def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])
    actions = torch.tensor(batch[1], dtype=torch.long)
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

# Vòng lặp mô phỏng
running = True
episode = 0

while running:
    state = env.reset()
    total_reward = 0
    for t in range(200):
        draw_environment()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Chọn hành động
        action = select_action(state_tensor, policy_net, epsilon, action_size)

        # Ánh xạ hành động
        action_mapping = [
            [0, -1],  # Lên
            [0, 1],   # Xuống
            [-1, 0],  # Trái
            [1, 0],   # Phải
            [0, 0]    # Giữ nguyên
        ]
        action_vector = action_mapping[action]

        # Thực hiện hành động
        next_state, reward, done = env.step(action_vector)
        total_reward += reward

        # Lưu kinh nghiệm
        memory.push(state, action, reward, next_state, done)

        # Tối ưu hóa
        optimize_model()

        # Cập nhật trạng thái
        state = next_state

        # Cập nhật màn hình
        pygame.display.flip()
        clock.tick(30)

        if done:
            break

    # Giảm epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Cập nhật mạng target
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total Reward: {total_reward}")
    episode += 1

    # Kiểm tra sự kiện thoát
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
