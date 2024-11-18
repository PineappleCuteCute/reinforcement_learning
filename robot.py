# File: robot.py
import pygame
import math
import json

class Robot:
    def __init__(self, start_x, start_y, size, color=(255, 0, 0)):#màu đỏ
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
        return math.hypot(target_x - self.rect.x, target_y - self.rect.y)

    def get_position(self):
        # Lấy vị trí hiện tại của robot
        return self.rect.x, self.rect.y

    def get_velocity(self):
        # Lấy vận tốc hiện tại của robot
        return self.velocity

    def save_state_to_file(self, position_filename="robot_position.json", velocity_filename="robot_velocity.json"):
        # Lưu vị trí của robot vào file JSON
        position_data = {
            "position": self.get_position()
        }
        with open(position_filename, 'w') as f:
            json.dump(position_data, f, indent=4)

        # Lưu vận tốc của robot vào file JSON
        velocity_data = {
            "velocity": self.get_velocity()
        }
        with open(velocity_filename, 'w') as f:
            json.dump(velocity_data, f, indent=4)



# # File: robot.py
# import pygame
# import math
# import json

# class Robot:
#     def __init__(self, start_x, start_y, size, color=(0, 255, 0)):
#         self.rect = pygame.Rect(start_x, start_y, size, size)
#         self.color = color
#         self.velocity = [0, 0]  # Vận tốc ban đầu (dx, dy)

#     def move(self):
#         # Di chuyển robot theo vận tốc hiện tại
#         self.rect.x += self.velocity[0]
#         self.rect.y += self.velocity[1]

#         # Giới hạn vị trí robot trong màn hình
#         self.rect.x = max(0, min(800 - self.rect.width, self.rect.x))
#         self.rect.y = max(0, min(600 - self.rect.height, self.rect.y))

#     def draw(self, screen):
#         # Vẽ robot lên màn hình
#         pygame.draw.rect(screen, self.color, self.rect)

#     def set_velocity(self, vx, vy):
#         # Đặt vận tốc mới cho robot
#         self.velocity = [vx, vy]

#     def calculate_distance(self, target_x, target_y):
#         # Tính khoảng cách từ robot tới một điểm (x, y)
#         return math.hypot(target_x - self.rect.x, target_y - self.rect.y)

#     def get_position(self):
#         # Lấy vị trí hiện tại của robot
#         return self.rect.x, self.rect.y

#     def get_velocity(self):
#         # Lấy vận tốc hiện tại của robot
#         return self.velocity

#     def save_state_to_file(self, filename="robot_state.json"):
#         # Lưu trạng thái của robot (vị trí và vận tốc) vào file JSON
#         data = {
#             "position": self.get_position(),
#             "velocity": self.get_velocity()
#         }
#         with open(filename, 'w') as f:
#             json.dump(data, f, indent=4)


# # # File: robot.py
# # import pygame
# # import math
# # import json

# # class Robot:
# #     def __init__(self, start_x, start_y, size, color=(0, 255, 0)):
# #         self.rect = pygame.Rect(start_x, start_y, size, size)
# #         self.color = color
# #         self.velocity = [0, 0]  # Vận tốc ban đầu (dx, dy)

# #     def move(self):
# #         # Di chuyển robot theo vận tốc hiện tại
# #         self.rect.x += self.velocity[0]
# #         self.rect.y += self.velocity[1]

# #         # Giới hạn vị trí robot trong màn hình
# #         self.rect.x = max(0, min(800 - self.rect.width, self.rect.x))
# #         self.rect.y = max(0, min(600 - self.rect.height, self.rect.y))

# #     def draw(self, screen):
# #         # Vẽ robot lên màn hình
# #         pygame.draw.rect(screen, self.color, self.rect)

# #     def set_velocity(self, vx, vy):
# #         # Đặt vận tốc mới cho robot
# #         self.velocity = [vx, vy]

# #     def calculate_distance(self, target_x, target_y):
# #         # Tính khoảng cách từ robot tới một điểm (x, y)
# #         return math.hypot(target_x - self.rect.x, target_y - self.rect.y)

# #     def get_position(self):
# #         # Lấy vị trí hiện tại của robot
# #         return self.rect.x, self.rect.y

# #     def get_velocity(self):
# #         # Lấy vận tốc hiện tại của robot
# #         return self.velocity

# #     def save_state_to_file(self, filename="robot_state.json"):
# #         # Lưu trạng thái của robot (vị trí và vận tốc) vào file JSON
# #         data = {
# #             "position": self.get_position(),
# #             "velocity": self.get_velocity()
# #         }
# #         with open(filename, 'w') as f:
# #             json.dump(data, f, indent=4)

# # # # File: robot.py
# # # #Robot Module Integration
# # # import pygame
# # # import math

# # # class Robot:
# # #     def __init__(self, start_x, start_y, size, color=(0, 255, 0)):
# # #         self.rect = pygame.Rect(start_x, start_y, size, size)
# # #         self.color = color
# # #         self.velocity = [0, 0]  # Vận tốc ban đầu (dx, dy)

# # #     def move(self):
# # #         # Di chuyển robot theo vận tốc hiện tại
# # #         self.rect.x += self.velocity[0]
# # #         self.rect.y += self.velocity[1]

# # #         # Giới hạn vị trí robot trong màn hình
# # #         self.rect.x = max(0, min(800 - self.rect.width, self.rect.x))
# # #         self.rect.y = max(0, min(600 - self.rect.height, self.rect.y))

# # #     def draw(self, screen):
# # #         # Vẽ robot lên màn hình
# # #         pygame.draw.rect(screen, self.color, self.rect)

# # #     def set_velocity(self, vx, vy):
# # #         # Đặt vận tốc mới cho robot
# # #         self.velocity = [vx, vy]

# # #     def calculate_distance(self, target_x, target_y):
# # #         # Tính khoảng cách từ robot tới một điểm (x, y)
# # #         return math.hypot(target_x - self.rect.x, target_y - self.rect.y)

# # #     def get_position(self):
# # #         # Lấy vị trí hiện tại của robot
# # #         return self.rect.x, self.rect.y

# # #     def get_velocity(self):
# # #         # Lấy vận tốc hiện tại của robot
# # #         return self.velocity