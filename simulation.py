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
pygame.display.set_caption('Môi trường với phản xạ động')

# Định nghĩa màu sắc
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Định nghĩa các thông số ô trong bản đồ
CELL_SIZE = 20
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Khởi tạo môi trường với một số chướng ngại vật tĩnh
tiled_map = np.zeros((ROWS, COLS))
static_obstacles = []

# Định nghĩa tường trong môi trường
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

# Định nghĩa chướng ngại vật động
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE - 5, CELL_SIZE - 5) for _ in range(20)]
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(20)]

# Khởi tạo Robot
robot = Robot(CELL_SIZE, CELL_SIZE, CELL_SIZE - 5)

# Định nghĩa điểm đích
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)

# Lưu đường đi của robot
robot_trail = []

# cách xử lý khi va chạm
def reflect_velocity(velocity, normal): #Tính vận tốc phản xạ dựa trên vector pháp tuyến.
    """Tính vận tốc phản xạ dựa trên vector pháp tuyến."""
    velocity = np.array(velocity)
    normal = np.array(normal)
    v_new = velocity - 2 * np.dot(velocity, normal) * normal
    return v_new.tolist()

# Thay vì cập nhật chướng ngại vật động trong vòng lặp chính, ta gọi hàm update_moving_obstacles
def update_moving_obstacles():
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
            if obs.colliderect(static_obs): #Sử dụng colliderect để kiểm tra va chạm.
                # Tính vector pháp tuyến
                normal = [0, 0]
                if abs(obs.right - static_obs.left) < 5:  # Va chạm từ bên phải
                    normal = [-1, 0]
                elif abs(obs.left - static_obs.right) < 5:  # Va chạm từ bên trái
                    normal = [1, 0]
                elif abs(obs.bottom - static_obs.top) < 5:  # Va chạm từ phía dưới
                    normal = [0, -1]
                elif abs(obs.top - static_obs.bottom) < 5:  # Va chạm từ phía trên
                    normal = [0, 1]

                # Tính vận tốc phản xạ
                new_velocity = reflect_velocity([dx, dy], normal)
                dx, dy = new_velocity

        # Cập nhật vị trí
        obstacle_directions[index] = (dx, dy)
        obs.x += dx * 5
        obs.y += dy * 5

# Vòng lặp chính
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # Vẽ môi trường
    for obs in static_obstacles:
        pygame.draw.rect(screen, BLUE, obs)
    for obs in moving_obstacles:
        pygame.draw.rect(screen, RED, obs)

    # Vẽ robot và điểm đích
    robot.draw(screen)
    pygame.draw.rect(screen, YELLOW, goal_point)

    # Vẽ đường đi của robot
    for trail in robot_trail:
        pygame.draw.circle(screen, GREEN, trail, 3)

    # Cập nhật vị trí chướng ngại vật động
    update_moving_obstacles()

    # Cập nhật vị trí robot
    robot.move()
    robot_trail.append(robot.get_position())

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
