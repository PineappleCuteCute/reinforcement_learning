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
CELL_SIZE = 20  # Kích thước mỗi ô của mê cung (giảm kích thước từ 40 xuống 20)
ROWS = SCREEN_HEIGHT // CELL_SIZE
COLS = SCREEN_WIDTH // CELL_SIZE

# Định nghĩa các bức tường ngoài cùng
outer_walls = [
    pygame.Rect(0, 0, SCREEN_WIDTH, CELL_SIZE),  # Tường trên cùng
    pygame.Rect(0, SCREEN_HEIGHT - CELL_SIZE, SCREEN_WIDTH, CELL_SIZE),  # Tường dưới cùng
    pygame.Rect(0, 0, CELL_SIZE, SCREEN_HEIGHT),  # Tường bên trái
    pygame.Rect(SCREEN_WIDTH - CELL_SIZE, 0, CELL_SIZE, SCREEN_HEIGHT)  # Tường bên phải
]

# Tạo mê cung tĩnh với các tường nối nhau bên trong mê cung
maze_walls = []
for i in range(1, ROWS-1):
    for j in range(1, COLS-1):
        if np.random.rand() < 0.1:  # Xác suất 10% để tạo các đoạn tường dài
            # Tạo các đoạn tường dài (nối nhiều ô lại với nhau)
            wall_length = np.random.randint(3, 6)  # Chiều dài ngẫu nhiên từ 3 đến 6 ô
            if np.random.rand() > 0.5:
                # Tạo tường theo chiều ngang
                if j + wall_length < COLS - 1:
                    maze_walls.append(pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, wall_length * CELL_SIZE, CELL_SIZE))
            else:
                # Tạo tường theo chiều dọc
                if i + wall_length < ROWS - 1:
                    maze_walls.append(pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, wall_length * CELL_SIZE))

# Định nghĩa 20 chướng ngại vật động (giảm kích thước từ 20x20 xuống 15x15)
moving_obstacles = [pygame.Rect(np.random.randint(1, COLS-1) * CELL_SIZE,
                                np.random.randint(1, ROWS-1) * CELL_SIZE,
                                CELL_SIZE - 5, CELL_SIZE - 5) for _ in range(20)]

# Hướng di chuyển của chướng ngại vật động
obstacle_directions = [(np.random.choice([-1, 1]), np.random.choice([-1, 1])) for _ in range(20)]

# Định nghĩa điểm bắt đầu và điểm đích
start_point = pygame.Rect(40, 40, CELL_SIZE - 5, CELL_SIZE - 5)  # Điểm bắt đầu (xanh lá cây)
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)  # Điểm đích (vàng)

# Hàm vẽ các bức tường ngoài cùng
def draw_outer_walls():
    for wall in outer_walls:
        pygame.draw.rect(screen, BLUE, wall)

# Hàm vẽ các tường nối trong mê cung
def draw_maze_walls():
    for wall in maze_walls:
        pygame.draw.rect(screen, BLUE, wall)

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
        new_rect = obs.move(dx * 5, dy * 5)

        # Kiểm tra nếu di chuyển có vượt qua tường ngoài hoặc chạm vào tường trong mê cung
        can_move = True

        # Kiểm tra nếu chạm vào tường ngoài cùng
        for wall in outer_walls:
            if new_rect.colliderect(wall):
                can_move = False
                break

        # Kiểm tra nếu chạm vào các tường nối trong mê cung
        if can_move:
            for wall in maze_walls:
                if new_rect.colliderect(wall):
                    can_move = False
                    break

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

    # Vẽ các tường ngoài cùng và tường nối trong mê cung
    draw_outer_walls()
    draw_maze_walls()

    # Vẽ chướng ngại vật động
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
