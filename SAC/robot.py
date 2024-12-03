import pygame
import torch
import numpy as np

class Robot:
    def __init__(self, x, y, size, sac_agent):
        # Vị trí và kích thước của robot
        self.x = x
        self.y = y
        self.size = size

        # Vận tốc của robot (di chuyển theo trục x và y)
        self.velocity_x = 0
        self.velocity_y = 0
        
        # SAC Agent cho hành động tự động
        self.sac_agent = sac_agent

    def move(self, action=None):
        """
        Di chuyển robot dựa trên hành động được chọn từ SAC.
        Nếu `action` được cung cấp, di chuyển robot theo [dx, dy].
        Nếu không, di chuyển dựa trên vận tốc hiện tại.
        """
        if action:
            # Di chuyển theo hành động
            self.x += action[0]
            self.y += action[1]

        # Giới hạn robot trong màn hình (800x600)
        self.x = max(self.size, min(self.x, 800 - self.size))
        self.y = max(self.size, min(self.y, 600 - self.size))

    def set_position(self, x, y):
        """Cập nhật vị trí của robot"""
        self.x = x
        self.y = y

    def get_position(self):
        """Trả về vị trí hiện tại của robot."""
        return [self.x, self.y]

    def draw(self, screen):
        """Vẽ robot lên màn hình."""
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

    def act(self, state):
        """
        Chọn hành động dựa trên SAC agent và di chuyển robot.
        """
        action = self.sac_agent.select_action(state)
        action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
        self.move(action_mapping[action])

    def move_to_goal(self, goal_x, goal_y, environment):
        """
        Di chuyển robot từ vị trí hiện tại đến điểm đích (goal_x, goal_y)
        mà không va chạm với các chướng ngại vật động và tĩnh.
        """
        state = self.get_state(environment)
        done = False
        while not done:
            # Chọn hành động và di chuyển
            self.act(state)
            state = self.get_state(environment)
            
            # Kiểm tra va chạm
            done = self.check_collision(environment)
            
            # Nếu đến đích, kết thúc
            if self.x == goal_x and self.y == goal_y:
                done = True
                print("Robot đã đến đích!")

    def check_collision(self, environment):
        """Kiểm tra va chạm với các chướng ngại vật động và tĩnh."""
        # Kiểm tra với các chướng ngại vật tĩnh
        robot_rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
        for obs in environment.static_obstacles:
            if robot_rect.colliderect(obs):
                print("Robot va chạm với chướng ngại vật tĩnh!")
                return True

        # Kiểm tra với các chướng ngại vật động
        for obs in environment.moving_obstacles:
            if robot_rect.colliderect(obs):
                print("Robot va chạm với chướng ngại vật động!")
                return True

        return False

    def get_state(self, environment):
        """
        Trả về trạng thái hiện tại của robot bao gồm vị trí và thông tin môi trường.
        """
        return np.array([self.x, self.y])  # Trạng thái là vị trí của robot




# import pygame

# class Robot:
#     def __init__(self, x, y, size):
#         # Vị trí và kích thước của robot
#         self.x = x
#         self.y = y
#         self.size = size
        
#         # Vận tốc của robot (di chuyển theo trục x và y)
#         self.velocity_x = 0
#         self.velocity_y = 0

#     def move(self, action=None):
#         """
#         Di chuyển robot.
#         Nếu `action` được cung cấp, di chuyển robot theo [dx, dy].
#         Nếu không, di chuyển dựa trên vận tốc hiện tại.
#         """
#         if action:
#             # Di chuyển theo hành động
#             self.x += action[0]
#             self.y += action[1]
#         else:
#             # Di chuyển theo vận tốc
#             self.x += self.velocity_x
#             self.y += self.velocity_y

#         # Giới hạn robot trong màn hình (800x600)
#         self.x = max(self.size, min(self.x, 800 - self.size))
#         self.y = max(self.size, min(self.y, 600 - self.size))

#     def set_velocity(self, velocity_x, velocity_y):
#         """Cập nhật vận tốc của robot."""
#         self.velocity_x = velocity_x
#         self.velocity_y = velocity_y

#     def get_position(self):
#         """Trả về vị trí hiện tại của robot."""
#         return [self.x, self.y]

#     def draw(self, screen):
#         """Vẽ robot lên màn hình."""
#         pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)
