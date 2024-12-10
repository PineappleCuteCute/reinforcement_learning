import pygame
import torch
import numpy as np
from robot import Robot
from environment import Environment
from dqn import DQN, select_action
import random
from sac_agent import SACAgent  # Nhập khẩu SAC agent nếu cần

# Khởi tạo và sử dụng môi trường trong simulation.py
from environment import Environment  # Giả sử lớp Environment nằm trong file environment.py

# Khởi tạo agent và môi trường
sac_agent = SACAgent(state_dim=4, action_dim=5)  # Đảm bảo tham số đúng với tên trong __init__ của SACAgent
# Khởi tạo agent với tham số tương ứng
env = Environment(width=800, height=600, sac_agent=sac_agent)  # Khởi tạo môi trường

# Các tham số cần thiết cho việc khởi tạo DQN
state_size = 4  # Ví dụ trạng thái có 4 chiều
action_size = 5  # Ví dụ có 5 hành động

# Khởi tạo DQN
policy_net = DQN(state_dim=state_size, action_dim=action_size)

# In ra cấu trúc của mô hình
print(policy_net)

# Khởi tạo Pygame
pygame.init()

# Thiết lập cửa sổ Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Mô phỏng môi trường động với phản xạ')

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Định nghĩa thông số bản đồ
CELL_SIZE = 20
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Khởi tạo môi trường với các chướng ngại vật tĩnh
tiled_map = np.zeros((ROWS, COLS))
static_obstacles = []

def create_open_map():
    """Tạo bản đồ với các chướng ngại vật tĩnh."""
    for _ in range(20):
        start_row = random.randint(1, ROWS - 2)
        start_col = random.randint(1, COLS - 2)
        length = random.randint(3, 8)
        if random.choice([True, False]):
            for i in range(length):
                if start_col + i < COLS - 1:
                    tiled_map[start_row][start_col + i] = 1
                    static_obstacles.append(pygame.Rect((start_col + i) * CELL_SIZE, start_row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        else:
            for i in range(length):
                if start_row + i < ROWS - 1:
                    tiled_map[start_row + i][start_col] = 1
                    static_obstacles.append(pygame.Rect(start_col * CELL_SIZE, (start_row + i) * CELL_SIZE, CELL_SIZE, CELL_SIZE))

create_open_map()

# Khởi tạo robot
robot = Robot(CELL_SIZE, CELL_SIZE, CELL_SIZE - 5)

# Điểm đích
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)

# Các tham số cần thiết cho việc khởi tạo DQN
state_dim = 4  # Ví dụ trạng thái có 4 chiều
action_dim = 5  # Ví dụ có 5 hành động

# Khởi tạo DQN
policy_net = DQN(state_dim=state_dim, action_dim=action_dim)

# In ra cấu trúc của mô hình
print(policy_net)

# policy_net.load_state_dict(torch.load("policy_net.pth"))
policy_net.eval()  # Chuyển mô hình về chế độ inference

# Hàm vẽ môi trường
def draw_environment():
    """Vẽ lại môi trường, robot, và các đối tượng khác."""
    screen.fill(WHITE)  # Làm mới màn hình
    pygame.draw.rect(screen, RED, (robot.x, robot.y, 50, 50))  # Vẽ robot
    pygame.draw.rect(screen, GREEN, (goal_point.x, goal_point.y, 50, 50))  # Vẽ mục tiêu
    pygame.display.update()  # Cập nhật giao diện

# Hàm kiểm tra va chạm
def check_collision():
    """Kiểm tra va chạm giữa robot và các chướng ngại vật."""
    robot_rect = pygame.Rect(robot.x - robot.size, robot.y - robot.size, robot.size * 2, robot.size * 2)

    # Kiểm tra va chạm với chướng ngại vật tĩnh
    for obs in static_obstacles:
        if robot_rect.colliderect(obs):
            return True

    # Kiểm tra va chạm với chướng ngại vật động
    return False

# Hàm di chuyển robot
# Hàm di chuyển robot
# def move_robot():
#     """Di chuyển robot từ điểm bắt đầu đến điểm đích."""

#     state = np.array([robot.x, robot.y, goal_point.x, goal_point.y])  # state có kích thước (4,)
#     print("State:", state)  # In ra giá trị của state để kiểm tra
#     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Chuyển thành (1, 4)
#     print("State tensor shape:", state_tensor.shape)  # Kiểm tra kích thước của state_tensor


#     state = np.array([robot.x, robot.y, goal_point.x, goal_point.y])  # state có kích thước (4,)
#     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Chuyển thành (1, 4)
    
#     done = False
#     total_reward = 0

#     while not done:
#         # Chọn hành động từ mô hình DQN
#         action = select_action(state_tensor, policy_net, epsilon=0.1, action_size=5)

#         # Mảng các hành động di chuyển (có thể mở rộng thêm các hành động)
#         action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
#         next_state, reward, done = env.step(action_mapping[action])  # Sử dụng trực tiếp action
#   # Thực hiện hành động và nhận trạng thái tiếp theo

#         # Cập nhật robot
#         robot.move(action_mapping[action.item()])

#         # Kiểm tra va chạm
#         if check_collision():
#             print("Robot va chạm với chướng ngại vật! Dừng chương trình.")
#             break

#         # Cập nhật trạng thái
#         state = next_state
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Cập nhật state_tensor

#         total_reward += reward  # Cộng dồn phần thưởng

#         # Vẽ lại môi trường sau mỗi bước
#         draw_environment()

#         # Kiểm tra nếu robot đã đến đích
#         if robot.x >= goal_point.x and robot.x <= goal_point.x + goal_point.width and \
#            robot.y >= goal_point.y and robot.y <= goal_point.y + goal_point.height:
#             print(f"Robot đã đến đích! Tổng phần thưởng: {total_reward}")
#             break

#         # Cập nhật trạng thái
#         state = next_state

#         total_reward += reward  # Cộng dồn phần thưởng

#         # Vẽ lại môi trường sau mỗi bước
#         draw_environment()

#         # Kiểm tra nếu robot đã đến đích
#         if robot.x >= goal_point.x and robot.x <= goal_point.x + goal_point.width and \
#            robot.y >= goal_point.y and robot.y <= goal_point.y + goal_point.height:
#             print(f"Robot đã đến đích! Tổng phần thưởng: {total_reward}")
#             break

# # Chạy mô phỏng di chuyển robot
# move_robot()

# Kết thúc Pygame
pygame.quit()
