import pygame
import numpy as np
import random
import json
from robot import Robot  # Import lớp Robot từ file robot.py

# Khởi tạo Pygame
pygame.init()

# Thiết lập cửa sổ Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Môi trường với chướng ngại vật tĩnh và động, điểm bắt đầu và đích')

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Định nghĩa các thông số ô trong bản đồ
CELL_SIZE = 20
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Khởi tạo môi trường với một số chướng ngại vật tĩnh
tiled_map = np.zeros((ROWS, COLS))
static_obstacles = []

# Định nghĩa một số tường trong môi trường
def create_open_map():
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

# Định nghĩa 20 chướng ngại vật động
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE - 5, CELL_SIZE - 5) for _ in range(20)]

# Hướng di chuyển của chướng ngại vật động
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(20)]

# Khởi tạo Robot từ lớp Robot đã định nghĩa
robot = Robot(CELL_SIZE, CELL_SIZE, CELL_SIZE - 5)

# Định nghĩa điểm đích
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)

# Hàm lưu vị trí của robot, chướng ngại vật tĩnh và động vào file JSON
def save_positions_to_file():
    positions_data = {
        "robot": {
            "position": robot.get_position()
        },
        "static_obstacles": [],
        "moving_obstacles": []
    }

    # Lưu tọa độ của chướng ngại vật tĩnh
    for obs in static_obstacles:
        positions_data["static_obstacles"].append({
            "x": int(obs.x),
            "y": int(obs.y),
            "width": int(obs.width),
            "height": int(obs.height)
        })

    # Lưu tọa độ của chướng ngại vật động
    for obs in moving_obstacles:
        positions_data["moving_obstacles"].append({
            "x": int(obs.x),
            "y": int(obs.y)
        })

    # Ghi dữ liệu vào file JSON
    with open('positions.json', 'w') as f:
        json.dump(positions_data, f, indent=4)

# Hàm lưu vận tốc của robot và chướng ngại vật động vào file JSON
def save_velocities_to_file():
    velocities_data = {
        "robot_velocity": robot.get_velocity(),
        "moving_obstacle_velocities": []
    }

    # Lưu vận tốc của chướng ngại vật động
    for dx, dy in obstacle_directions:
        velocities_data["moving_obstacle_velocities"].append({
            "dx": int(dx),
            "dy": int(dy)
        })

    # Ghi dữ liệu vào file JSON
    with open('velocities.json', 'w') as f:
        json.dump(velocities_data, f, indent=4)

# Vòng lặp chính
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # Vẽ môi trường và chướng ngại vật động
    for obs in static_obstacles:
        pygame.draw.rect(screen, BLUE, obs)
    for obs in moving_obstacles:
        pygame.draw.rect(screen, RED, obs)

    # Vẽ điểm bắt đầu (robot) và điểm đích
    robot.draw(screen)
    pygame.draw.rect(screen, YELLOW, goal_point)

    # Cập nhật vị trí chướng ngại vật động
    for index, obs in enumerate(moving_obstacles):
        dx, dy = obstacle_directions[index]
        new_x = obs.x + dx * 5
        new_y = obs.y + dy * 5

        can_move = True

        if new_x < CELL_SIZE or new_x + obs.width > SCREEN_WIDTH - CELL_SIZE or new_y < CELL_SIZE or new_y + obs.height > SCREEN_HEIGHT - CELL_SIZE:
            can_move = False

        if can_move:
            new_col = new_x // CELL_SIZE
            new_row = new_y // CELL_SIZE
            if tiled_map[new_row][new_col] == 1:
                can_move = False

        if can_move:
            obs.x = new_x
            obs.y = new_y
        else:
            obstacle_directions[index] = (-dx, -dy)

    # Cập nhật vị trí robot
    robot.move()

    # Lưu trạng thái của robot và các chướng ngại vật
    save_positions_to_file()
    save_velocities_to_file()

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

    # Cập nhật màn hình
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
