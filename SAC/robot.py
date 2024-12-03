import pygame
import numpy as np

class Robot:
    def __init__(self, x, y, size, sac_agent=None):
        # Vị trí và kích thước của robot
        self.x = x
        self.y = y
        self.size = size
        self.sac_agent = sac_agent

        # Tạo hình chữ nhật đại diện cho robot
        self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    def move(self, action=None):
        """
        Di chuyển robot dựa trên hành động được chọn.
        """
        if action:
            # Di chuyển theo hành động [dx, dy]
            self.x += action[0]
            self.y += action[1]

        # Giới hạn robot trong màn hình (800x600)
        self.x = max(self.size, min(self.x, 800 - self.size))
        self.y = max(self.size, min(self.y, 600 - self.size))

        # Cập nhật lại rect sau khi di chuyển
        self.rect.x = self.x - self.size
        self.rect.y = self.y - self.size

    def set_position(self, x, y):
        """Cập nhật vị trí của robot"""
        self.x = x
        self.y = y
        self.rect.x = self.x - self.size
        self.rect.y = self.y - self.size

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
        if self.sac_agent:
            action = self.sac_agent.select_action(state)  # Lấy hành động từ SAC agent
            # Giả sử action là một số nguyên tương ứng với các hành động
            action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
            if isinstance(action, int) and 0 <= action < len(action_mapping):
                self.move(action_mapping[action])
            else:
                print("Hành động không hợp lệ:", action)
        else:
            # Nếu không có SAC agent, robot không di chuyển
            pass

    def check_collision(self, environment):
        """Kiểm tra va chạm với các chướng ngại vật động và tĩnh."""
        for obs in environment.static_obstacles:
            if self.rect.colliderect(obs):  # Kiểm tra va chạm với chướng ngại vật tĩnh
                print("Robot va chạm với chướng ngại vật tĩnh!")
                return True

        # Kiểm tra với các chướng ngại vật động
        for obs in environment.moving_obstacles:
            if self.rect.colliderect(obs):  # Kiểm tra va chạm với chướng ngại vật động
                print("Robot va chạm với chướng ngại vật động!")
                return True

        return False

    def get_state(self, environment):
        """
        Trả về trạng thái hiện tại của robot bao gồm vị trí và thông tin môi trường.
        """
        return np.array([self.x, self.y])  # Trạng thái là vị trí của robot



# # robot.py
# import pygame
# import numpy as np

# class Robot:
#     def __init__(self, x, y, size, sac_agent=None):
#         # Vị trí và kích thước của robot
#         self.x = x
#         self.y = y
#         self.size = size

#         # SAC Agent cho hành động tự động
#         self.sac_agent = sac_agent

#     def move(self, action=None):
#         """
#         Di chuyển robot theo hành động được chọn.
#         """
#         if action:
#             self.x += action[0]
#             self.y += action[1]

#         # Giới hạn robot trong màn hình (800x600)
#         self.x = max(self.size, min(self.x, 800 - self.size))
#         self.y = max(self.size, min(self.y, 600 - self.size))

#     def set_position(self, x, y):
#         """Cập nhật vị trí của robot."""
#         self.x = x
#         self.y = y

#     def get_position(self):
#         """Trả về vị trí hiện tại của robot."""
#         return [self.x, self.y]

#     def draw(self, screen):
#         """Vẽ robot lên màn hình."""
#         pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

#     def act(self, state):
#         """
#         Chọn hành động dựa trên SAC agent và di chuyển robot.
#         """
#         if self.sac_agent:
#             action = self.sac_agent.select_action(state)
#             action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
#             self.move(action_mapping[action])


# # import pygame
# # import torch
# # import numpy as np

# # class Robot:
# #     def __init__(self, x, y, size, sac_agent):
# #         # Vị trí và kích thước của robot
# #         self.x = x
# #         self.y = y
# #         self.size = size

# #         # Vận tốc của robot (di chuyển theo trục x và y)
# #         self.velocity_x = 0
# #         self.velocity_y = 0
        
# #         # SAC Agent cho hành động tự động
# #         self.sac_agent = sac_agent

# #         # Khởi tạo rect cho robot
# #         self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

# #     def move(self, action=None):
# #         """
# #         Di chuyển robot dựa trên hành động được chọn từ SAC.
# #         Nếu `action` được cung cấp, di chuyển robot theo [dx, dy].
# #         Nếu không, di chuyển dựa trên vận tốc hiện tại.
# #         """
# #         if action:
# #             # Di chuyển theo hành động (dx, dy)
# #             self.x += action[0]
# #             self.y += action[1]

# #         # Giới hạn robot trong màn hình (800x600)
# #         self.x = max(self.size, min(self.x, 800 - self.size))
# #         self.y = max(self.size, min(self.y, 600 - self.size))

# #     def set_position(self, x, y):
# #         """Cập nhật vị trí của robot"""
# #         self.x = x
# #         self.y = y

# #     def get_position(self):
# #         """Trả về vị trí hiện tại của robot."""
# #         return [self.x, self.y]

# #     def draw(self, screen):
# #         """Vẽ robot lên màn hình."""
# #         pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

# #     def act(self, state):
# #         """
# #         Chọn hành động dựa trên SAC agent và di chuyển robot.
# #         """
# #         action = self.sac_agent.select_action(state)  # Lấy hành động từ SAC agent
# #         action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # Các hành động: [up, down, left, right, stay]
# #         self.move(action_mapping[action])

# #     def move_to_goal(self, goal_x, goal_y, environment):
# #         """
# #         Di chuyển robot từ vị trí hiện tại đến điểm đích (goal_x, goal_y)
# #         mà không va chạm với các chướng ngại vật động và tĩnh.
# #         """
# #         state = self.get_state(environment)  # Lấy trạng thái hiện tại của robot
# #         done = False
# #         while not done:
# #             # Chọn hành động và di chuyển
# #             self.act(state)
# #             state = self.get_state(environment)  # Cập nhật trạng thái sau khi di chuyển
            
# #             # Kiểm tra va chạm
# #             done = self.check_collision(environment)
            
# #             # Nếu đến đích, kết thúc
# #             if self.x == goal_x and self.y == goal_y:
# #                 done = True
# #                 print("Robot đã đến đích!")

# #     def check_collision(self, environment):
# #         """Kiểm tra va chạm với các chướng ngại vật động và tĩnh."""
# #         robot_rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
        
# #         # Kiểm tra với các chướng ngại vật tĩnh
# #         for obs in environment.static_obstacles:
# #             if robot_rect.colliderect(obs):
# #                 print("Robot va chạm với chướng ngại vật tĩnh!")
# #                 return True

# #         # Kiểm tra với các chướng ngại vật động
# #         for obs in environment.moving_obstacles:
# #             if robot_rect.colliderect(obs):
# #                 print("Robot va chạm với chướng ngại vật động!")
# #                 return True

# #         return False

# #     def get_state(self, environment):
# #         """
# #         Trả về trạng thái hiện tại của robot bao gồm vị trí và thông tin môi trường.
# #         """
# #         # Lấy trạng thái robot + thông tin về các chướng ngại vật (tĩnh và động)
# #         state = np.array([self.x, self.y])
# #         return state



# # # import pygame
# # # import torch
# # # import numpy as np

# # # class Robot:
# # #     def __init__(self, x, y, size, sac_agent):
# # #         # Vị trí và kích thước của robot
# # #         self.x = x
# # #         self.y = y
# # #         self.size = size

# # #         # Vận tốc của robot (di chuyển theo trục x và y)
# # #         self.velocity_x = 0
# # #         self.velocity_y = 0
        
# # #         # SAC Agent cho hành động tự động
# # #         self.sac_agent = sac_agent

# # #     def move(self, action=None):
# # #         """
# # #         Di chuyển robot dựa trên hành động được chọn từ SAC.
# # #         Nếu `action` được cung cấp, di chuyển robot theo [dx, dy].
# # #         Nếu không, di chuyển dựa trên vận tốc hiện tại.
# # #         """
# # #         if action:
# # #             # Di chuyển theo hành động
# # #             self.x += action[0]
# # #             self.y += action[1]

# # #         # Giới hạn robot trong màn hình (800x600)
# # #         self.x = max(self.size, min(self.x, 800 - self.size))
# # #         self.y = max(self.size, min(self.y, 600 - self.size))

# # #     def set_position(self, x, y):
# # #         """Cập nhật vị trí của robot"""
# # #         self.x = x
# # #         self.y = y

# # #     def get_position(self):
# # #         """Trả về vị trí hiện tại của robot."""
# # #         return [self.x, self.y]

# # #     def draw(self, screen):
# # #         """Vẽ robot lên màn hình."""
# # #         pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)

# # #     def act(self, state):
# # #         """
# # #         Chọn hành động dựa trên SAC agent và di chuyển robot.
# # #         """
# # #         action = self.sac_agent.select_action(state)
# # #         action_mapping = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # [up, down, left, right, stay]
# # #         self.move(action_mapping[action])

# # #     def move_to_goal(self, goal_x, goal_y, environment):
# # #         """
# # #         Di chuyển robot từ vị trí hiện tại đến điểm đích (goal_x, goal_y)
# # #         mà không va chạm với các chướng ngại vật động và tĩnh.
# # #         """
# # #         state = self.get_state(environment)
# # #         done = False
# # #         while not done:
# # #             # Chọn hành động và di chuyển
# # #             self.act(state)
# # #             state = self.get_state(environment)
            
# # #             # Kiểm tra va chạm
# # #             done = self.check_collision(environment)
            
# # #             # Nếu đến đích, kết thúc
# # #             if self.x == goal_x and self.y == goal_y:
# # #                 done = True
# # #                 print("Robot đã đến đích!")

# # #     def check_collision(self, environment):
# # #         """Kiểm tra va chạm với các chướng ngại vật động và tĩnh."""
# # #         # Kiểm tra với các chướng ngại vật tĩnh
# # #         robot_rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
# # #         for obs in environment.static_obstacles:
# # #             if robot_rect.colliderect(obs):
# # #                 print("Robot va chạm với chướng ngại vật tĩnh!")
# # #                 return True

# # #         # Kiểm tra với các chướng ngại vật động
# # #         for obs in environment.moving_obstacles:
# # #             if robot_rect.colliderect(obs):
# # #                 print("Robot va chạm với chướng ngại vật động!")
# # #                 return True

# # #         return False

# # #     def get_state(self, environment):
# # #         """
# # #         Trả về trạng thái hiện tại của robot bao gồm vị trí và thông tin môi trường.
# # #         """
# # #         return np.array([self.x, self.y])  # Trạng thái là vị trí của robot




# # # # import pygame

# # # # class Robot:
# # # #     def __init__(self, x, y, size):
# # # #         # Vị trí và kích thước của robot
# # # #         self.x = x
# # # #         self.y = y
# # # #         self.size = size
        
# # # #         # Vận tốc của robot (di chuyển theo trục x và y)
# # # #         self.velocity_x = 0
# # # #         self.velocity_y = 0

# # # #     def move(self, action=None):
# # # #         """
# # # #         Di chuyển robot.
# # # #         Nếu `action` được cung cấp, di chuyển robot theo [dx, dy].
# # # #         Nếu không, di chuyển dựa trên vận tốc hiện tại.
# # # #         """
# # # #         if action:
# # # #             # Di chuyển theo hành động
# # # #             self.x += action[0]
# # # #             self.y += action[1]
# # # #         else:
# # # #             # Di chuyển theo vận tốc
# # # #             self.x += self.velocity_x
# # # #             self.y += self.velocity_y

# # # #         # Giới hạn robot trong màn hình (800x600)
# # # #         self.x = max(self.size, min(self.x, 800 - self.size))
# # # #         self.y = max(self.size, min(self.y, 600 - self.size))

# # # #     def set_velocity(self, velocity_x, velocity_y):
# # # #         """Cập nhật vận tốc của robot."""
# # # #         self.velocity_x = velocity_x
# # # #         self.velocity_y = velocity_y

# # # #     def get_position(self):
# # # #         """Trả về vị trí hiện tại của robot."""
# # # #         return [self.x, self.y]

# # # #     def draw(self, screen):
# # # #         """Vẽ robot lên màn hình."""
# # # #         pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.size)
