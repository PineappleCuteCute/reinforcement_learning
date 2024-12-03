import pygame
import random
import numpy as np
from robot import Robot 
from sac_agent import SACAgent  # Nhập khẩu SAC agent nếu cần

class Environment:
    def __init__(self, width, height, cell_size=20, sac_agent=None):
        # Kích thước màn hình
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.rows = height // cell_size
        self.cols = width // cell_size

        # Khởi tạo các chướng ngại vật tĩnh
        self.static_obstacles = []
        self.create_static_obstacles()

        # Khởi tạo các chướng ngại vật động
        self.moving_obstacles = []
        self.create_moving_obstacles()

        # Đặt mục tiêu (goal) cho robot
        self.goal = pygame.Rect(width - 2 * cell_size, height - 2 * cell_size, cell_size - 5, cell_size - 5)

        # Khởi tạo robot nếu sac_agent được cung cấp
        if sac_agent:
            self.robot = Robot(self.cols // 2 * self.cell_size, self.rows // 2 * self.cell_size, size=10, sac_agent=sac_agent)
        else:
            self.robot = None  # Bạn có thể khởi tạo robot sau nếu cần

    def create_static_obstacles(self):
        """Tạo các chướng ngại vật tĩnh (có thể là các hình chữ nhật hoặc hình vuông)"""
        for _ in range(10):  # Tạo 10 chướng ngại vật tĩnh
            start_row = random.randint(1, self.rows - 2)
            start_col = random.randint(1, self.cols - 2)
            length = random.randint(3, 8)
            direction = random.choice(['horizontal', 'vertical'])

            if direction == 'horizontal':
                for i in range(length):
                    if start_col + i < self.cols - 1:
                        rect = pygame.Rect((start_col + i) * self.cell_size, start_row * self.cell_size, self.cell_size, self.cell_size)
                        self.static_obstacles.append(rect)
            else:
                for i in range(length):
                    if start_row + i < self.rows - 1:
                        rect = pygame.Rect(start_col * self.cell_size, (start_row + i) * self.cell_size, self.cell_size, self.cell_size)
                        self.static_obstacles.append(rect)

    def create_moving_obstacles(self):
        """Tạo các chướng ngại vật động (các hình vuông nhỏ sẽ di chuyển ngẫu nhiên)"""
        for _ in range(5):  # Tạo 5 chướng ngại vật động
            x = random.randint(1, self.cols - 2) * self.cell_size
            y = random.randint(1, self.rows - 2) * self.cell_size
            moving_rect = pygame.Rect(x, y, self.cell_size - 5, self.cell_size - 5)
            self.moving_obstacles.append(moving_rect)

        # Hướng di chuyển ngẫu nhiên cho chướng ngại vật động
        self.obstacle_directions = [(random.choice([-1, 1]), random.choice([-1, 1])) for _ in range(len(self.moving_obstacles))]

    def update_moving_obstacles(self):
        """Cập nhật vị trí và hướng của các chướng ngại vật động"""
        for index, obs in enumerate(self.moving_obstacles):
            dx, dy = self.obstacle_directions[index]
            new_x = obs.x + dx * 5
            new_y = obs.y + dy * 5

            # Kiểm tra va chạm với biên màn hình
            if new_x < self.cell_size or new_x + obs.width > self.width - self.cell_size:
                dx = -dx
            if new_y < self.cell_size or new_y + obs.height > self.height - self.cell_size:
                dy = -dy

            # Cập nhật hướng di chuyển
            self.obstacle_directions[index] = (dx, dy)

            # Cập nhật vị trí chướng ngại vật động
            obs.x += dx * 5
            obs.y += dy * 5

    def reset(self):
        """Khởi tạo lại môi trường và trả về trạng thái ban đầu."""
        self.static_obstacles.clear()
        self.create_static_obstacles()

        self.moving_obstacles.clear()
        self.create_moving_obstacles()

        # Trạng thái ban đầu là vị trí của robot (có thể điều chỉnh thêm nếu cần)
        initial_state = np.array([self.cols // 2 * self.cell_size, self.rows // 2 * self.cell_size])
        return initial_state

    def render(self, screen):
        """Vẽ môi trường lên màn hình."""
        screen.fill((255, 255, 255))  # Màu nền trắng

        # Vẽ các chướng ngại vật tĩnh
        for obs in self.static_obstacles:
            pygame.draw.rect(screen, (0, 0, 0), obs)  # Màu đen cho chướng ngại vật tĩnh

        # Vẽ các chướng ngại vật động
        for obs in self.moving_obstacles:
            pygame.draw.rect(screen, (255, 0, 0), obs)  # Màu đỏ cho chướng ngại vật động

        pygame.display.flip()  # Cập nhật màn hình

    def step(self, action):
        """
        Thực hiện một hành động và trả về trạng thái tiếp theo, phần thưởng và trạng thái kết thúc.
        Action: [dx, dy] (di chuyển robot theo các hướng)
        """
        # Di chuyển robot theo hành động
        robot_rect = pygame.Rect(self.robot.x - self.robot.size, self.robot.y - self.robot.size, self.robot.size * 2, self.robot.size * 2)
        robot_rect.x += action[0]
        robot_rect.y += action[1]

        # Cập nhật vị trí robot và kiểm tra va chạm
        self.robot.set_position(robot_rect.x, robot_rect.y)
        done = self.robot.check_collision(self)

        # Cập nhật vị trí các chướng ngại vật động
        self.update_moving_obstacles()

        # Tính phần thưởng
        reward = -1  # Phạt mỗi lần di chuyển
        if done:
            reward = -100  # Phạt nếu va chạm
        elif self.robot.rect.colliderect(self.goal):
            reward = 100  # Thưởng nếu đến đích

        # Trạng thái tiếp theo
        next_state = np.array([self.robot.x, self.robot.y])

        return next_state, reward, done



# import pygame
# import random
# import numpy as np

# class Environment:
#     def __init__(self, width, height, cell_size=20):
#         # Kích thước màn hình
#         self.width = width
#         self.height = height
#         self.cell_size = cell_size
#         self.rows = height // cell_size
#         self.cols = width // cell_size

#         # Khởi tạo các chướng ngại vật tĩnh
#         self.static_obstacles = []
#         self.create_static_obstacles()

#         # Khởi tạo các chướng ngại vật động
#         self.moving_obstacles = []
#         self.create_moving_obstacles()

#     def create_static_obstacles(self):
#         """Tạo các chướng ngại vật tĩnh (có thể là các hình chữ nhật hoặc hình vuông)"""
#         for _ in range(10):  # Tạo 10 chướng ngại vật tĩnh
#             start_row = random.randint(1, self.rows - 2)
#             start_col = random.randint(1, self.cols - 2)
#             length = random.randint(3, 8)
#             direction = random.choice(['horizontal', 'vertical'])

#             if direction == 'horizontal':
#                 for i in range(length):
#                     if start_col + i < self.cols - 1:
#                         rect = pygame.Rect((start_col + i) * self.cell_size, start_row * self.cell_size, self.cell_size, self.cell_size)
#                         self.static_obstacles.append(rect)
#             else:
#                 for i in range(length):
#                     if start_row + i < self.rows - 1:
#                         rect = pygame.Rect(start_col * self.cell_size, (start_row + i) * self.cell_size, self.cell_size, self.cell_size)
#                         self.static_obstacles.append(rect)

#     def create_moving_obstacles(self):
#         """Tạo các chướng ngại vật động (các hình vuông nhỏ sẽ di chuyển ngẫu nhiên)"""
#         for _ in range(5):  # Tạo 5 chướng ngại vật động
#             x = random.randint(1, self.cols - 2) * self.cell_size
#             y = random.randint(1, self.rows - 2) * self.cell_size
#             moving_rect = pygame.Rect(x, y, self.cell_size - 5, self.cell_size - 5)
#             self.moving_obstacles.append(moving_rect)

#         # Hướng di chuyển ngẫu nhiên cho chướng ngại vật động
#         self.obstacle_directions = [(random.choice([-1, 1]), random.choice([-1, 1])) for _ in range(len(self.moving_obstacles))]

#     def update_moving_obstacles(self):
#         """Cập nhật vị trí và hướng của các chướng ngại vật động"""
#         for index, obs in enumerate(self.moving_obstacles):
#             dx, dy = self.obstacle_directions[index]
#             new_x = obs.x + dx * 5
#             new_y = obs.y + dy * 5

#             # Kiểm tra va chạm với biên màn hình
#             if new_x < self.cell_size or new_x + obs.width > self.width - self.cell_size:
#                 dx = -dx
#             if new_y < self.cell_size or new_y + obs.height > self.height - self.cell_size:
#                 dy = -dy

#             # Cập nhật hướng di chuyển
#             self.obstacle_directions[index] = (dx, dy)

#             # Cập nhật vị trí chướng ngại vật động
#             obs.x += dx * 5
#             obs.y += dy * 5

#     def reset(self):
#         """Khởi tạo lại môi trường và trả về trạng thái ban đầu."""
#         self.static_obstacles.clear()
#         self.create_static_obstacles()

#         self.moving_obstacles.clear()
#         self.create_moving_obstacles()

#         # Trạng thái ban đầu là vị trí của robot (có thể điều chỉnh thêm nếu cần)
#         initial_state = np.array([self.cols // 2 * self.cell_size, self.rows // 2 * self.cell_size])
#         return initial_state

#     def render(self, screen):
#         """Vẽ môi trường lên màn hình."""
#         screen.fill((255, 255, 255))  # Màu nền trắng

#         # Vẽ các chướng ngại vật tĩnh
#         for obs in self.static_obstacles:
#             pygame.draw.rect(screen, (0, 0, 0), obs)  # Màu đen cho chướng ngại vật tĩnh

#         # Vẽ các chướng ngại vật động
#         for obs in self.moving_obstacles:
#             pygame.draw.rect(screen, (255, 0, 0), obs)  # Màu đỏ cho chướng ngại vật động

#         pygame.display.flip()  # Cập nhật màn hình

#     def step(self, action):
#         """
#         Thực hiện một hành động và trả về trạng thái tiếp theo, phần thưởng và trạng thái kết thúc.
#         Action: [dx, dy] (di chuyển robot theo các hướng)
#         """
#         # Di chuyển robot theo hành động
#         robot_rect = pygame.Rect(self.robot.x - self.robot.size, self.robot.y - self.robot.size, self.robot.size * 2, self.robot.size * 2)
#         robot_rect.x += action[0]
#         robot_rect.y += action[1]

#         # Cập nhật vị trí robot và kiểm tra va chạm
#         self.robot.set_position(robot_rect.x, robot_rect.y)
#         done = self.robot.check_collision(self)

#         # Cập nhật vị trí các chướng ngại vật động
#         self.update_moving_obstacles()

#         # Tính phần thưởng (Ví dụ: -1 nếu va chạm, +1 nếu đạt được mục tiêu)
#         reward = -1 if done else 0  # Cụ thể hóa phần thưởng tùy theo yêu cầu

#         return self.get_state(), reward, done

#     def get_state(self):
#         """
#         Trả về trạng thái hiện tại của môi trường (ví dụ: vị trí robot, các chướng ngại vật)
#         """
#         return np.array([self.robot.x, self.robot.y])  # Trạng thái hiện tại có thể bao gồm vị trí của robot, v.v.





# # import numpy as np
# # import random
# # import pygame
# # from robot import Robot  # Lớp robot mà bạn đã tạo trước đó

# # class Environment:
# #     def __init__(self, width, height, robot_start, goal_position, static_obstacles, moving_obstacles):
# #         self.width = width
# #         self.height = height
# #         self.robot = Robot(*robot_start, size=20)
# #         self.goal = goal_position
# #         self.static_obstacles = static_obstacles
# #         self.moving_obstacles = moving_obstacles
# #         self.robot_trail = []

# #     def reset(self):
# #         """Khôi phục lại trạng thái ban đầu của môi trường."""
# #         self.robot.x, self.robot.y = self.robot.x, self.robot.y
# #         self.robot_trail = []
# #         return self.get_state()

# #     def get_state(self):
# #         """Trả về trạng thái hiện tại của robot."""
# #         state = [self.robot.x, self.robot.y, self.goal.x, self.goal.y]
# #         # Thêm vị trí chướng ngại vật tĩnh và động vào trạng thái
# #         state.extend([obs.x for obs in self.static_obstacles])
# #         state.extend([obs.y for obs in self.static_obstacles])
# #         state.extend([obs.x for obs in self.moving_obstacles])
# #         state.extend([obs.y for obs in self.moving_obstacles])
# #         return np.array(state)

# #     def step(self, action):
# #         """Thực hiện hành động, tính toán phần thưởng, và cập nhật trạng thái."""
# #         # Di chuyển robot theo hành động
# #         self.robot.move(action)
        
# #         # Cập nhật các chướng ngại vật động
# #         self.update_moving_obstacles()

# #         # Tính phần thưởng
# #         reward = self.calculate_reward()

# #         # Kiểm tra xem robot đã đến đích chưa
# #         done = self.check_goal_reached()

# #         # Trả về trạng thái, phần thưởng, và điều kiện kết thúc
# #         return self.get_state(), reward, done

# #     def update_moving_obstacles(self):
# #         """Cập nhật vị trí chướng ngại vật động."""
# #         for obs in self.moving_obstacles:
# #             # Di chuyển ngẫu nhiên
# #             obs.x += random.choice([-1, 1]) * 5
# #             obs.y += random.choice([-1, 1]) * 5

# #     def calculate_reward(self):
# #         """Tính toán phần thưởng."""
# #         distance_to_goal = np.linalg.norm([self.robot.x - self.goal.x, self.robot.y - self.goal.y])
        
# #         # Phạt nếu va chạm với chướng ngại vật
# #         if self.check_collision():
# #             return -10
        
# #         # Thưởng nếu robot đến gần mục tiêu
# #         if distance_to_goal < 10:
# #             return 10
        
# #         # Nếu không thì phạt theo khoảng cách
# #         return -distance_to_goal

# #     def check_collision(self):
# #         """Kiểm tra va chạm với chướng ngại vật."""
# #         robot_rect = pygame.Rect(self.robot.x - self.robot.size, self.robot.y - self.robot.size, self.robot.size * 2, self.robot.size * 2)
        
# #         for obs in self.static_obstacles + self.moving_obstacles:
# #             if robot_rect.colliderect(obs):
# #                 return True
# #         return False

# #     def check_goal_reached(self):
# #         """Kiểm tra robot đã đến đích chưa."""
# #         return np.linalg.norm([self.robot.x - self.goal.x, self.robot.y - self.goal.y]) < 10


# # # import numpy as np
# # # import random
# # # from robot import Robot
# # # import pygame

# # # class Environment:
# # #     def __init__(self, width, height, num_dynamic_obs=5, num_static_obs=5):
# # #         self.width = width
# # #         self.height = height
# # #         self.robot = Robot(x=width // 2, y=height // 2, size=20)
# # #         self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic_obs)
# # #         self.static_obstacles = self._create_static_obstacles(num_static_obs)
# # #         self.goal = [width - 40, height - 40]

# # #     def _create_dynamic_obstacles(self, num):
# # #         obstacles = []
# # #         for _ in range(num):
# # #             x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
# # #             size = random.randint(10, 20)
# # #             velocity = [random.choice([-1, 1]), random.choice([-1, 1])]
# # #             obstacles.append({'position': [x, y], 'size': size, 'velocity': velocity})
# # #         return obstacles

# # #     def _create_static_obstacles(self, num):
# # #         obstacles = []
# # #         for _ in range(num):
# # #             x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
# # #             size = random.randint(20, 50)
# # #             obstacles.append({'position': [x, y], 'size': size})
# # #         return obstacles

# # #     def step(self, action):
# # #         """Thực hiện một bước mô phỏng."""
# # #         # Tính toán vị trí dự kiến
# # #         proposed_x = self.robot.x + action[0]
# # #         proposed_y = self.robot.y + action[1]

# # #         # Kiểm tra va chạm trước khi di chuyển
# # #         if not self._check_proposed_collision(proposed_x, proposed_y):
# # #             self.robot.move(action)  # Di chuyển nếu không va chạm
# # #         else:
# # #             print("Robot không thể di chuyển vì có va chạm.")

# # #         # Cập nhật vị trí chướng ngại vật động
# # #         self._update_moving_obstacles()

# # #         # Kiểm tra va chạm sau khi di chuyển
# # #         done = self._check_collision()

# # #         # Tính phần thưởng và kiểm tra kết thúc
# # #         reward = self._calculate_reward(done)

# # #         return self._get_state(), reward, done

# # #     def _update_moving_obstacles(self):
# # #         for obs in self.dynamic_obstacles:
# # #             dx, dy = obs['velocity']
# # #             obs['position'][0] += dx * 5
# # #             obs['position'][1] += dy * 5
# # #             if obs['position'][0] < 0 or obs['position'][0] > self.width:
# # #                 obs['velocity'][0] = -dx
# # #             if obs['position'][1] < 0 or obs['position'][1] > self.height:
# # #                 obs['velocity'][1] = -dy

# # #     def _check_collision(self):
# # #         """Kiểm tra va chạm giữa robot và các chướng ngại vật hiện tại."""
# # #         rx, ry = self.robot.get_position()
# # #         for obs in self.dynamic_obstacles + self.static_obstacles:
# # #             ox, oy = obs['position']
# # #             distance = np.linalg.norm([rx - ox, ry - oy])
# # #             if distance < (self.robot.size + obs['size']) / 2:
# # #                 return True  # Va chạm
# # #         return False  # Không có va chạm

# # #     def _check_proposed_collision(self, x, y):
# # #         for obs in self.dynamic_obstacles + self.static_obstacles:
# # #             ox, oy = obs['position']
# # #             distance = np.linalg.norm([x - ox, y - oy])
# # #             if distance < (self.robot.size + obs['size']) / 2:
# # #                 return True
# # #         return False

# # #     def _calculate_reward(self, collision):
# # #         if collision:
# # #             return -100  # Phạt nếu va chạm
# # #         goal_distance = np.linalg.norm([self.robot.x - self.goal[0], self.robot.y - self.goal[1]])
# # #         return -goal_distance  # Khuyến khích robot di chuyển về phía mục tiêu

# # #     def _get_state(self):
# # #         state = list(self.robot.get_position())
# # #         for obs in self.dynamic_obstacles:
# # #             state.extend(obs['position'])
# # #             state.extend(obs['velocity'])
# # #         for obs in self.static_obstacles:
# # #             state.extend(obs['position'])
# # #         state.extend(self.goal)
# # #         return np.array(state, dtype=np.float32)

# # #     def reset(self):
# # #         self.robot = Robot(x=self.width // 2, y=self.height // 2, size=20)
# # #         self.dynamic_obstacles = self._create_dynamic_obstacles(5)
# # #         self.static_obstacles = self._create_static_obstacles(5)
# # #         return self._get_state()
