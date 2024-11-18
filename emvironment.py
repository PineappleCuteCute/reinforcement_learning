import pygame
import numpy as np

# Khởi tạo Pygame
pygame.init()

# Thiết lập cửa sổ Pygame
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Mê cung với chướng ngại vật tĩnh và động')

# Định nghĩa màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

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

# Định nghĩa chướng ngại vật động
moving_obstacles = [pygame.Rect(100, 100, CELL_SIZE, CELL_SIZE),
                    pygame.Rect(300, 300, CELL_SIZE, CELL_SIZE)]

# Hướng di chuyển của chướng ngại vật động
obstacle_directions = [(1, 0), (0, 1)]  # (dx, dy) cho mỗi chướng ngại vật

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

        # Cập nhật vị trí
        obs.x += dx * 5
        obs.y += dy * 5

        # Đảo ngược hướng khi gặp tường hoặc vượt quá biên
        if obs.left < 0 or obs.right > SCREEN_WIDTH:
            obstacle_directions[index] = (-dx, dy)
        if obs.top < 0 or obs.bottom > SCREEN_HEIGHT:
            obstacle_directions[index] = (dx, -dy)

# Vòng lặp chính
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)  # Làm sạch màn hình với màu trắng

    # Vẽ mê cung và chướng ngại vật động
    draw_maze()
    draw_moving_obstacles()

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
