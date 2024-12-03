import numpy as np
import random
from robot import Robot
import pygame

class Environment:
    def __init__(self, width, height, num_dynamic_obs=5, num_static_obs=5):
        self.width = width
        self.height = height
        self.robot = Robot(x=width // 2, y=height // 2, size=20)
        self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic_obs)
        self.static_obstacles = self._create_static_obstacles(num_static_obs)
        self.goal = [width - 40, height - 40]

    def _create_dynamic_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(10, 20)
            velocity = [random.choice([-1, 1]), random.choice([-1, 1])]
            obstacles.append({'position': [x, y], 'size': size, 'velocity': velocity})
        return obstacles

    def _create_static_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(20, 50)
            obstacles.append({'position': [x, y], 'size': size})
        return obstacles

    def step(self, action):
        """Thực hiện một bước mô phỏng."""
        # Tính toán vị trí dự kiến
        proposed_x = self.robot.x + action[0]
        proposed_y = self.robot.y + action[1]

        # Kiểm tra va chạm trước khi di chuyển
        if not self._check_proposed_collision(proposed_x, proposed_y):
            self.robot.move(action)  # Di chuyển nếu không va chạm
        else:
            print("Robot không thể di chuyển vì có va chạm.")

        # Cập nhật vị trí chướng ngại vật động
        self._update_moving_obstacles()

        # Kiểm tra va chạm sau khi di chuyển
        done = self._check_collision()

        # Tính phần thưởng và kiểm tra kết thúc
        reward = self._calculate_reward(done)

        return self._get_state(), reward, done

    def _update_moving_obstacles(self):
        for obs in self.dynamic_obstacles:
            dx, dy = obs['velocity']
            obs['position'][0] += dx * 5
            obs['position'][1] += dy * 5
            if obs['position'][0] < 0 or obs['position'][0] > self.width:
                obs['velocity'][0] = -dx
            if obs['position'][1] < 0 or obs['position'][1] > self.height:
                obs['velocity'][1] = -dy

    def _check_collision(self):
        """Kiểm tra va chạm giữa robot và các chướng ngại vật hiện tại."""
        rx, ry = self.robot.get_position()
        for obs in self.dynamic_obstacles + self.static_obstacles:
            ox, oy = obs['position']
            distance = np.linalg.norm([rx - ox, ry - oy])
            if distance < (self.robot.size + obs['size']) / 2:
                return True  # Va chạm
        return False  # Không có va chạm

    def _check_proposed_collision(self, x, y):
        for obs in self.dynamic_obstacles + self.static_obstacles:
            ox, oy = obs['position']
            distance = np.linalg.norm([x - ox, y - oy])
            if distance < (self.robot.size + obs['size']) / 2:
                return True
        return False

    def _calculate_reward(self, collision):
        if collision:
            return -100  # Phạt nếu va chạm
        goal_distance = np.linalg.norm([self.robot.x - self.goal[0], self.robot.y - self.goal[1]])
        return -goal_distance  # Khuyến khích robot di chuyển về phía mục tiêu

    def _get_state(self):
        state = list(self.robot.get_position())
        for obs in self.dynamic_obstacles:
            state.extend(obs['position'])
            state.extend(obs['velocity'])
        for obs in self.static_obstacles:
            state.extend(obs['position'])
        state.extend(self.goal)
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.robot = Robot(x=self.width // 2, y=self.height // 2, size=20)
        self.dynamic_obstacles = self._create_dynamic_obstacles(5)
        self.static_obstacles = self._create_static_obstacles(5)
        return self._get_state()
