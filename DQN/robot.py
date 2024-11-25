import pygame

class Robot:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.velocity_x = 0
        self.velocity_y = 0

    def move(self, action=None):
        """
        Di chuyển robot.
        Nếu `action` được cung cấp, di chuyển theo [dx, dy].
        Nếu không, di chuyển dựa trên vận tốc hiện tại.
        """
        if action:
            self.x += action[0]
            self.y += action[1]
        else:
            self.x += self.velocity_x
            self.y += self.velocity_y

        # Giới hạn robot trong màn hình
        self.x = max(self.size, min(self.x, 800 - self.size))
        self.y = max(self.size, min(self.y, 600 - self.size))

    def set_velocity(self, velocity_x, velocity_y):
        """Cập nhật vận tốc của robot."""
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def get_position(self):
        """Trả về vị trí hiện tại của robot."""
        return [self.x, self.y]

    def draw(self, screen):
        """Vẽ robot lên màn hình."""
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)
