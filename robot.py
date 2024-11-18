# File: robot.py
# Robot Module Integration
import pygame
import math

class Robot:
    def __init__(self, start_x, start_y, size, color=(0, 255, 0)):
        self.rect = pygame.Rect(start_x, start_y, size, size)
        self.color = color
        self.velocity = [0, 0]  # Vận tốc ban đầu (dx, dy)

    def move(self):
        # Di chuyển robot theo vận tốc hiện tại
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

        # Giới hạn vị trí robot trong màn hình
        self.rect.x = max(0, min(800 - self.rect.width, self.rect.x))
        self.rect.y = max(0, min(600 - self.rect.height, self.rect.y))

    def draw(self, screen):
        # Vẽ robot lên màn hình
        pygame.draw.rect(screen, self.color, self.rect)

    def set_velocity(self, vx, vy):
        # Đặt vận tốc mới cho robot
        self.velocity = [vx, vy]

    def calculate_distance(self, target_x, target_y):
        # Tính khoảng cách từ robot tới một điểm (x, y)
        return math.sqrt((target_x - self.rect.x) ** 2 + (target_y - self.rect.y) ** 2)

    def get_position(self):
        # Lấy vị trí hiện tại của robot
        return self.rect.x, self.rect.y

    def get_velocity(self):
        # Lấy vận tốc hiện tại của robot
        return self.velocity
