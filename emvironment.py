import pygame
import numpy as np

# Khởi tạo Pygame
pygame.init()

# Thiết lập cửa sổ Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Mê cung với chướng ngại vật tĩnh và động, điểm bắt đầu và đích')

# Định nghĩa màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Định nghĩa các thông số mê cung
CELL_SIZE = 40  # Kích thước mỗi ô của mê cung
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Tạo mê cung tĩnh với các tường
maze = np.zeros((ROWS, COLS))
# Ví dụ: Đặt các bức tường trong mê cung
for i in range(ROWS):
    maze[i][0] = 1  # Tường dọc bên trái
    maze[i][COLS-1] = 1  # Tường dọc bên phải
for j in range(COLS):
    maze[0][j] = 1  # Tường ngang trên cùng
    maze[ROWS-1][j] = 1  # Tường ngang dưới cùng

# Thêm một số tường ngẫu nhiên ở giữa mê cung
for i in range(1, ROWS-1):
    for j in range(1, COLS-1):
        if np.random.rand() < 0.2:  # Xác suất 20% đặt tường
            maze[i][j] = 1

# Định nghĩa 20 chướng ngại vật động
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE, CELL_SIZE) for _ in range(20)]

# Hướng di chuyển của chướng ngại vật động
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(20)]

# Định nghĩa điểm bắt đầu và điểm đích
start_point = pygame.Rect(40, 40, CELL_SIZE, CELL_SIZE)  # Điểm bắt đầu (xanh lá cây)
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE, CELL_SIZE)  # Điểm đích (vàng)

# Hàm vẽ mê cung
def draw_maze():
    for row in range(ROWS):
        for col in range(COLS):
            if maze[row][col] == 1:
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

        # Tính toán ô mới của chướng ngại vật trong mê cung
        new_col = new_x // CELL_SIZE
        new_row = new_y // CELL_SIZE

        # Kiểm tra nếu di chuyển có vượt qua tường
        if 0 <= new_col < COLS and 0 <= new_row < ROWS and maze[new_row][new_col] == 0:
            # Nếu không vượt qua tường, cập nhật vị trí
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

    # Vẽ mê cung và chướng ngại vật động
    draw_maze()
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
