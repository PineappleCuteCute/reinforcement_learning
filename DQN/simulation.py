import pygame
import csv
import torch
import numpy as np
from dqn import DQN, select_action
from environment import Environment

# Thông số Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 30

# Thông số DQN
state_size = 8  # Số lượng trạng thái cần kiểm tra (đảm bảo tương ứng với `env._get_state()`)
action_size = 5  # Số hành động: [Lên, Xuống, Trái, Phải, Giữ nguyên]
policy_net_path = "policy_net.pth"  # Mô hình đã huấn luyện

# Khởi tạo Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Robot Simulation')
clock = pygame.time.Clock()

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Tải mô hình DQN đã huấn luyện
policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load(policy_net_path))
policy_net.eval()

# Khởi tạo môi trường
env = Environment(SCREEN_WIDTH, SCREEN_HEIGHT)

# Lưu dữ liệu vào CSV
csv_filename = "simulation_data.csv"
fields = ['time', 'robot_x', 'robot_y', 'goal_x', 'goal_y', 'reward', 'collision']
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

# Hàm lưu dữ liệu vào file CSV
def save_to_csv(data):
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

# Vòng lặp mô phỏng
running = True
state = env.reset()
time_step = 0
while running:
    screen.fill(WHITE)

    # Vẽ chướng ngại vật
    for obs in env.static_obstacles:
        pygame.draw.rect(screen, BLUE, (obs['position'][0], obs['position'][1], 20, 20))
    for obs in env.dynamic_obstacles:
        pygame.draw.rect(screen, RED, (obs['position'][0], obs['position'][1], 15, 15))

    # Vẽ robot
    robot_pos = env.robot.get_position()
    pygame.draw.circle(screen, GREEN, (int(robot_pos[0]), int(robot_pos[1])), env.robot.size)

    # Vẽ mục tiêu
    pygame.draw.rect(screen, YELLOW, (env.goal[0], env.goal[1], 20, 20))

    # Chọn hành động từ policy_net
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = select_action(state_tensor, policy_net, epsilon=0, action_size=action_size)
    action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]

    # Thực hiện hành động và cập nhật môi trường
    next_state, reward, collision = env.step(action_mapping[action])

    # Lưu dữ liệu vào CSV
    data = [
        time_step,  # Thời gian
        robot_pos[0], robot_pos[1],  # Vị trí robot
        env.goal[0], env.goal[1],  # Vị trí mục tiêu
        reward,  # Reward
        int(collision)  # Va chạm
    ]
    save_to_csv(data)

    # Kiểm tra sự kiện thoát
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Kiểm tra đạt mục tiêu
    if env.goal[0] - 20 <= robot_pos[0] <= env.goal[0] + 20 and env.goal[1] - 20 <= robot_pos[1] <= env.goal[1] + 20:
        print("Robot đạt mục tiêu!")
        running = False

    # Cập nhật màn hình
    pygame.display.flip()
    clock.tick(FPS)
    time_step += 1

    # Cập nhật trạng thái
    state = next_state

pygame.quit()
print(f"Dữ liệu mô phỏng đã được lưu vào {csv_filename}")
