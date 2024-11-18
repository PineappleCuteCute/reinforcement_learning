import pygame
import numpy as np
import random

# Khởi tạo Pygame
pygame.init()

# Thiết lập cửa sổ Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Môi trường với chướng ngại vật tĩnh và động, điểm bắt đầu và đích')

# Định nghĩa màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Định nghĩa các thông số ô trong bản đồ
CELL_SIZE = 20  # Kích thước mỗi ô
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Khởi tạo môi trường với một số chướng ngại vật tĩnh
tiled_map = np.zeros((ROWS, COLS))

# Định nghĩa một số tường trong môi trường
def create_open_map():
    for _ in range(20):  # Tạo 20 tường ngẫu nhiên
        start_row = random.randint(1, ROWS - 2)
        start_col = random.randint(1, COLS - 2)
        length = random.randint(3, 8)
        if random.choice([True, False]):
            # Tạo tường theo chiều ngang
            for i in range(length):
                if start_col + i < COLS - 1:
                    tiled_map[start_row][start_col + i] = 1
        else:
            # Tạo tường theo chiều dọc
            for i in range(length):
                if start_row + i < ROWS - 1:
                    tiled_map[start_row + i][start_col] = 1

# Tạo bản đồ mở với một số chướng ngại vật tĩnh
create_open_map()

# Định nghĩa 20 chướng ngại vật động (giảm kích thước từ 20x20 xuống 15x15)
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE - 5, CELL_SIZE - 5) for _ in range(20)]

# Hướng di chuyển của chướng ngại vật động
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(20)]

# Định nghĩa điểm bắt đầu và điểm đích
start_point = pygame.Rect(CELL_SIZE, CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)  # Điểm bắt đầu (xanh lá cây)
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)  # Điểm đích (vàng)

# Hàm vẽ môi trường với các chướng ngại vật tĩnh
def draw_map():
    for row in range(ROWS):
        for col in range(COLS):
            if tiled_map[row][col] == 1:
                pygame.draw.rect(screen, BLUE, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Hàm vẽ chướng ngại vật động
def draw_moving_obstacles():
    for obs in moving_obstacles:
        pygame.draw.rect(screen, RED, obs)

# Hàm cập nhật vị trí chướng ngại vật động
def update_moving_obstacles():
    for index, obs in enumerate(moving_obstacles):
        dx, dy = obstacle_directions[index]

        # Cập nhật vị trí tiềm năng mới
        new_x = obs.x + dx * 5
        new_y = obs.y + dy * 5

        # Tính toán ô mới của chướng ngại vật trong bản đồ
        new_rect = obs.move(dx * 5, dy * 5)

        # Kiểm tra nếu di chuyển có vượt qua tường ngoài hoặc chạm vào tường trong bản đồ
        can_move = True

        # Kiểm tra nếu chạm vào tường ngoài cùng
        if new_x < CELL_SIZE or new_x + obs.width > SCREEN_WIDTH - CELL_SIZE or new_y < CELL_SIZE or new_y + obs.height > SCREEN_HEIGHT - CELL_SIZE:
            can_move = False

        # Kiểm tra nếu chạm vào các tường trong bản đồ
        if can_move:
            new_col = new_x // CELL_SIZE
            new_row = new_y // CELL_SIZE
            if tiled_map[new_row][new_col] == 1:
                can_move = False

        # Nếu không chạm vào tường, cập nhật vị trí
        if can_move:
            obs.x = new_x
            obs.y = new_y
        else:
            # Nếu gặp tường, đổi hướng
            obstacle_directions[index] = (-dx, -dy)

# Vòng lặp chính
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)  # Làm sạch màn hình với màu trắng

    # Vẽ môi trường và chướng ngại vật động
    draw_map()
    draw_moving_obstacles()

    # Vẽ điểm bắt đầu và điểm đích
    pygame.draw.rect(screen, GREEN, start_point)
    pygame.draw.rect(screen, YELLOW, goal_point)

    # Cập nhật vị trí chướng ngại vật động
    update_moving_obstacles()

    # Kiểm tra sự kiện
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Cập nhật màn hình
    pygame.display.flip()

    # Giới hạn tốc độ khung hình
    clock.tick(30)

# Thoát Pygame
pygame.quit()
