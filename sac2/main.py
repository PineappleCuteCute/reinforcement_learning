import pygame
import random
import numpy as np

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
tiled_map = np.zeros((ROWS, COLS))  # Bản đồ ban đầu
static_obstacles = []  # Danh sách chướng ngại vật tĩnh

def create_open_map():
    """Tạo bản đồ với các chướng ngại vật tĩnh."""
    for _ in range(20):  # Tạo 20 chướng ngại vật ngẫu nhiên
        start_row = random.randint(1, ROWS - 2)
        start_col = random.randint(1, COLS - 2)
        length = random.randint(3, 8)
        if random.choice([True, False]):
            # Tạo đường chướng ngại vật ngang
            for i in range(length):
                if start_col + i < COLS - 1:
                    tiled_map[start_row][start_col + i] = 1
                    static_obstacles.append(pygame.Rect((start_col + i) * CELL_SIZE, start_row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        else:
            # Tạo đường chướng ngại vật dọc
            for i in range(length):
                if start_row + i < ROWS - 1:
                    tiled_map[start_row + i][start_col] = 1
                    static_obstacles.append(pygame.Rect(start_col * CELL_SIZE, (start_row + i) * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Tạo bản đồ
create_open_map()

# Khởi tạo robot
robot_x, robot_y = 100, 100  # Vị trí robot ban đầu

# Điểm đích
goal_point = pygame.Rect(SCREEN_WIDTH - 2 * CELL_SIZE, SCREEN_HEIGHT - 2 * CELL_SIZE, CELL_SIZE - 5, CELL_SIZE - 5)

# Hàm vẽ môi trường
def draw_environment():
    """Vẽ lại môi trường, robot, và các đối tượng khác."""
    screen.fill(WHITE)  # Làm mới màn hình

    # Vẽ các chướng ngại vật
    for obs in static_obstacles:
        pygame.draw.rect(screen, BLUE, obs)  # Màu xanh cho chướng ngại vật

    # Vẽ robot
    pygame.draw.rect(screen, RED, (robot_x, robot_y, 50, 50))  # Robot màu đỏ

    # Vẽ điểm đích
    pygame.draw.rect(screen, GREEN, (goal_point.x, goal_point.y, 50, 50))  # Điểm đích màu xanh lá

    pygame.display.update()  # Cập nhật giao diện

# Hàm kiểm tra va chạm
def check_collision():
    """Kiểm tra va chạm giữa robot và các chướng ngại vật."""
    robot_rect = pygame.Rect(robot_x, robot_y, 50, 50)

    # Kiểm tra va chạm với chướng ngại vật tĩnh
    for obs in static_obstacles:
        if robot_rect.colliderect(obs):
            return True

    # Kiểm tra va chạm với điểm đích
    if robot_rect.colliderect(goal_point):
        return "goal"

    return False

# Hàm di chuyển robot
def move_robot():
    """Di chuyển robot từ điểm bắt đầu đến điểm đích."""
    global robot_x, robot_y

    # Giả lập việc di chuyển robot
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Giả sử robot di chuyển về phía trước (ở đây chỉ di chuyển ngang, có thể mở rộng thêm)
        robot_x += 5

        # Kiểm tra va chạm
        collision = check_collision()
        if collision == "goal":
            print("Robot đã đến đích!")
            done = True
        elif collision:
            print("Robot va chạm với chướng ngại vật! Dừng lại.")
            done = True

        # Vẽ lại môi trường
        draw_environment()

# Chạy mô phỏng di chuyển robot
move_robot()

# Kết thúc Pygame
pygame.quit()
