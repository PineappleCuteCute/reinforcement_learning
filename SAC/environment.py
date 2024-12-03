import numpy as np
import random
import pygame
from robot import Robot  # Lớp robot mà bạn đã tạo trước đó

class Environment:
    def __init__(self, width, height, robot_start, goal_position, static_obstacles, moving_obstacles):
        self.width = width
        self.height = height
        self.robot = Robot(*robot_start, size=20)
        self.goal = goal_position
        self.static_obstacles = static_obstacles
        self.moving_obstacles = moving_obstacles
        self.robot_trail = []

    def reset(self):
        """Khôi phục lại trạng thái ban đầu của môi trường."""
        self.robot.x, self.robot.y = self.robot.x, self.robot.y
        self.robot_trail = []
        return self.get_state()

    def get_state(self):
        """Trả về trạng thái hiện tại của robot."""
        state = [self.robot.x, self.robot.y, self.goal.x, self.goal.y]
        # Thêm vị trí chướng ngại vật tĩnh và động vào trạng thái
        state.extend([obs.x for obs in self.static_obstacles])
        state.extend([obs.y for obs in self.static_obstacles])
        state.extend([obs.x for obs in self.moving_obstacles])
        state.extend([obs.y for obs in self.moving_obstacles])
        return np.array(state)

    def step(self, action):
        """Thực hiện hành động, tính toán phần thưởng, và cập nhật trạng thái."""
        # Di chuyển robot theo hành động
        self.robot.move(action)
        
        # Cập nhật các chướng ngại vật động
        self.update_moving_obstacles()

        # Tính phần thưởng
        reward = self.calculate_reward()

        # Kiểm tra xem robot đã đến đích chưa
        done = self.check_goal_reached()

        # Trả về trạng thái, phần thưởng, và điều kiện kết thúc
        return self.get_state(), reward, done

    def update_moving_obstacles(self):
        """Cập nhật vị trí chướng ngại vật động."""
        for obs in self.moving_obstacles:
            # Di chuyển ngẫu nhiên
            obs.x += random.choice([-1, 1]) * 5
            obs.y += random.choice([-1, 1]) * 5

    def calculate_reward(self):
        """Tính toán phần thưởng."""
        distance_to_goal = np.linalg.norm([self.robot.x - self.goal.x, self.robot.y - self.goal.y])
        
        # Phạt nếu va chạm với chướng ngại vật
        if self.check_collision():
            return -10
        
        # Thưởng nếu robot đến gần mục tiêu
        if distance_to_goal < 10:
            return 10
        
        # Nếu không thì phạt theo khoảng cách
        return -distance_to_goal

    def check_collision(self):
        """Kiểm tra va chạm với chướng ngại vật."""
        robot_rect = pygame.Rect(self.robot.x - self.robot.size, self.robot.y - self.robot.size, self.robot.size * 2, self.robot.size * 2)
        
        for obs in self.static_obstacles + self.moving_obstacles:
            if robot_rect.colliderect(obs):
                return True
        return False

    def check_goal_reached(self):
        """Kiểm tra robot đã đến đích chưa."""
        return np.linalg.norm([self.robot.x - self.goal.x, self.robot.y - self.goal.y]) < 10


# import numpy as np
# import random
# from robot import Robot
# import pygame

# class Environment:
#     def __init__(self, width, height, num_dynamic_obs=5, num_static_obs=5):
#         self.width = width
#         self.height = height
#         self.robot = Robot(x=width // 2, y=height // 2, size=20)
#         self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic_obs)
#         self.static_obstacles = self._create_static_obstacles(num_static_obs)
#         self.goal = [width - 40, height - 40]

#     def _create_dynamic_obstacles(self, num):
#         obstacles = []
#         for _ in range(num):
#             x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
#             size = random.randint(10, 20)
#             velocity = [random.choice([-1, 1]), random.choice([-1, 1])]
#             obstacles.append({'position': [x, y], 'size': size, 'velocity': velocity})
#         return obstacles

#     def _create_static_obstacles(self, num):
#         obstacles = []
#         for _ in range(num):
#             x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
#             size = random.randint(20, 50)
#             obstacles.append({'position': [x, y], 'size': size})
#         return obstacles

#     def step(self, action):
#         """Thực hiện một bước mô phỏng."""
#         # Tính toán vị trí dự kiến
#         proposed_x = self.robot.x + action[0]
#         proposed_y = self.robot.y + action[1]

#         # Kiểm tra va chạm trước khi di chuyển
#         if not self._check_proposed_collision(proposed_x, proposed_y):
#             self.robot.move(action)  # Di chuyển nếu không va chạm
#         else:
#             print("Robot không thể di chuyển vì có va chạm.")

#         # Cập nhật vị trí chướng ngại vật động
#         self._update_moving_obstacles()

#         # Kiểm tra va chạm sau khi di chuyển
#         done = self._check_collision()

#         # Tính phần thưởng và kiểm tra kết thúc
#         reward = self._calculate_reward(done)

#         return self._get_state(), reward, done

#     def _update_moving_obstacles(self):
#         for obs in self.dynamic_obstacles:
#             dx, dy = obs['velocity']
#             obs['position'][0] += dx * 5
#             obs['position'][1] += dy * 5
#             if obs['position'][0] < 0 or obs['position'][0] > self.width:
#                 obs['velocity'][0] = -dx
#             if obs['position'][1] < 0 or obs['position'][1] > self.height:
#                 obs['velocity'][1] = -dy

#     def _check_collision(self):
#         """Kiểm tra va chạm giữa robot và các chướng ngại vật hiện tại."""
#         rx, ry = self.robot.get_position()
#         for obs in self.dynamic_obstacles + self.static_obstacles:
#             ox, oy = obs['position']
#             distance = np.linalg.norm([rx - ox, ry - oy])
#             if distance < (self.robot.size + obs['size']) / 2:
#                 return True  # Va chạm
#         return False  # Không có va chạm

#     def _check_proposed_collision(self, x, y):
#         for obs in self.dynamic_obstacles + self.static_obstacles:
#             ox, oy = obs['position']
#             distance = np.linalg.norm([x - ox, y - oy])
#             if distance < (self.robot.size + obs['size']) / 2:
#                 return True
#         return False

#     def _calculate_reward(self, collision):
#         if collision:
#             return -100  # Phạt nếu va chạm
#         goal_distance = np.linalg.norm([self.robot.x - self.goal[0], self.robot.y - self.goal[1]])
#         return -goal_distance  # Khuyến khích robot di chuyển về phía mục tiêu

#     def _get_state(self):
#         state = list(self.robot.get_position())
#         for obs in self.dynamic_obstacles:
#             state.extend(obs['position'])
#             state.extend(obs['velocity'])
#         for obs in self.static_obstacles:
#             state.extend(obs['position'])
#         state.extend(self.goal)
#         return np.array(state, dtype=np.float32)

#     def reset(self):
#         self.robot = Robot(x=self.width // 2, y=self.height // 2, size=20)
#         self.dynamic_obstacles = self._create_dynamic_obstacles(5)
#         self.static_obstacles = self._create_static_obstacles(5)
#         return self._get_state()
