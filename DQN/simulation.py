import pygame
import csv
import numpy as np
import random
from robot import Robot  # Import lớp Robot từ file robot.py

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

# Khởi tạo chướng ngại vật động
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE - 5, CELL_SIZE - 5) for _ in range(10)]
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(10)]

# Khởi tạo robot
robot = Robot(CELL_SIZE, CELL_SIZE, CELL_SIZE - 5)

# Điểm đích
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)

# Lưu đường đi của robot
robot_trail = []

# Ghi dữ liệu ra CSV
csv_filename = "simulation_data.csv"
fields = ['time_step', 'robot_x', 'robot_y', 'goal_x', 'goal_y', 'collision']
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fields)

def save_to_csv(data):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Hàm kiểm tra va chạm
def check_collision(): #robot không va chạm vao chướng ngại vật!!!!
    """Kiểm tra va chạm giữa robot và các chướng ngại vật."""
    robot_rect = pygame.Rect(robot.x - robot.size, robot.y - robot.size, robot.size * 2, robot.size * 2)

    # Kiểm tra va chạm với chướng ngại vật tĩnh
    for obs in static_obstacles:
        if robot_rect.colliderect(obs):
            print("Robot đã va chạm với chướng ngại vật tĩnh! Dừng chương trình.")
            pygame.quit()
            exit(1)

    # Kiểm tra va chạm với chướng ngại vật động
    for obs in moving_obstacles:
        if robot_rect.colliderect(obs):
            print("Robot đã va chạm với chướng ngại vật động! Dừng chương trình.")
            pygame.quit()
            exit(1)

# Hàm cập nhật vị trí chướng ngại vật động
def update_moving_obstacles(): #tranning?, chướng ngại vật đông tĩnh lao vào nhau !!!!
    #tạo vector lộ trình di chuyển cho chướng ngại vật động, nẩy phản xạ, khi nào -dx -dy
    """Cập nhật vị trí và hướng của chướng ngại vật động."""
    for index, obs in enumerate(moving_obstacles):
        dx, dy = obstacle_directions[index]
        new_x = obs.x + dx * 5
        new_y = obs.y + dy * 5

        # Kiểm tra va chạm với biên
        if new_x < CELL_SIZE or new_x + obs.width > SCREEN_WIDTH - CELL_SIZE:
            dx = -dx
        if new_y < CELL_SIZE or new_y + obs.height > SCREEN_HEIGHT - CELL_SIZE:
            dy = -dy

        # Kiểm tra va chạm với chướng ngại vật tĩnh
        for static_obs in static_obstacles:
            if obs.colliderect(static_obs):
                # Phản xạ theo trục X hoặc Y dựa trên va chạm
                overlap_x = min(static_obs.right - obs.left, obs.right - static_obs.left)
                overlap_y = min(static_obs.bottom - obs.top, obs.bottom - static_obs.top)

                if overlap_x < overlap_y:  # Va chạm theo trục X
                    dx = -dx
                else:  # Va chạm theo trục Y
                    dy = -dy

        # Cập nhật hướng vận tốc
        obstacle_directions[index] = (dx, dy)

        # Cập nhật vị trí
        obs.x += dx * 5
        obs.y += dy * 5

# Vòng lặp chính
running = True
clock = pygame.time.Clock()
time_step = 0

while running:
    screen.fill(WHITE)

    # Vẽ chướng ngại vật
    for obs in static_obstacles:
        pygame.draw.rect(screen, BLUE, obs)
    for obs in moving_obstacles:
        pygame.draw.rect(screen, RED, obs)

    # Vẽ robot và điểm đích
    robot.draw(screen)
    pygame.draw.rect(screen, YELLOW, goal_point)

    # Vẽ đường đi của robot
    for trail in robot_trail:
        pygame.draw.circle(screen, GREEN, trail, 2)

    # Cập nhật vị trí chướng ngại vật động
    update_moving_obstacles()

    # Cập nhật vị trí robot
    robot.move()
    robot_trail.append(robot.get_position())

    # Kiểm tra va chạm
    check_collision()

    # Ghi dữ liệu vào CSV
    collision = any([robot.get_position()[0] == obs.x and robot.get_position()[1] == obs.y for obs in moving_obstacles])
    save_to_csv([time_step, robot.get_position()[0], robot.get_position()[1], goal_point.x, goal_point.y, int(collision)])

    # Kiểm tra sự kiện
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                robot.set_velocity(0, -5)
            elif event.key == pygame.K_DOWN:
                robot.set_velocity(0, 5)
            elif event.key == pygame.K_LEFT:
                robot.set_velocity(-5, 0)
            elif event.key == pygame.K_RIGHT:
                robot.set_velocity(5, 0)

    # Kiểm tra va chạm với điểm đích
    if goal_point.collidepoint(robot.get_position()):
        print("Robot đã đạt được điểm đích!")
        running = False

    # Cập nhật màn hình
    pygame.display.flip()
    clock.tick(30)
    time_step += 1

pygame.quit()
print(f"Dữ liệu mô phỏng đã được lưu vào {csv_filename}")
