# robot.py

import pygame

class Robot:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.velocity_x = 0
        self.velocity_y = 0

    def move(self):
        """Cập nhật vị trí robot dựa trên vận tốc."""
        self.x += self.velocity_x
        self.y += self.velocity_y

        # Giới hạn trong màn hình
        self.x = max(self.size, min(self.x, 800 - self.size))
        self.y = max(self.size, min(self.y, 600 - self.size))

    def set_velocity(self, velocity_x, velocity_y):
        """Cài đặt vận tốc cho robot."""
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def get_position(self):
        """Trả về vị trí hiện tại của robot."""
        return int(self.x), int(self.y)

    def get_velocity(self):
        """Trả về vận tốc hiện tại của robot."""
        return self.velocity_x, self.velocity_y

    def draw(self, screen):
        """Vẽ robot lên màn hình."""
        pygame.draw.circle(screen, (0, 0, 255), (int(self.x), int(self.y)), self.size)
