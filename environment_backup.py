import numpy as np
import random
from robot import Robot

class Environment:
    def __init__(self, width, height, num_dynamic_obs=5, num_static_obs=5):
        self.width = width
        self.height = height
        self.robot = Robot(x=width // 2, y=height // 2, size=20)  # Robot tại trung tâm
        self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic_obs)
        self.static_obstacles = self._create_static_obstacles(num_static_obs)
        self.goal = [width - 40, height - 40]  # Mục tiêu của robot

    def _create_dynamic_obstacles(self, num):
        """Khởi tạo chướng ngại vật động."""
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(10, 20)
            velocity = [random.choice([-1, 1]), random.choice([-1, 1])]
            obstacles.append({'position': [x, y], 'size': size, 'velocity': velocity})
        return obstacles

    def _create_static_obstacles(self, num):
        """Khởi tạo chướng ngại vật tĩnh."""
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(20, 50)
            obstacles.append({'position': [x, y], 'size': size})
        return obstacles

    def step(self, action):
        """
        Thực hiện một bước mô phỏng:
        - Action là vector [dx, dy] cho robot.
        """
        self.robot.move(action)

        # Cập nhật chướng ngại vật động
        for obs in self.dynamic_obstacles:
            x, y = obs['position']
            dx, dy = obs['velocity']
            new_x, new_y = x + dx * 2, y + dy * 2

            # Phản xạ khi va chạm biên
            if new_x < 0 or new_x > self.width:
                dx = -dx
            if new_y < 0 or new_y > self.height:
                dy = -dy

            # Phản xạ khi va chạm chướng ngại vật tĩnh
            for static_obs in self.static_obstacles:
                sx, sy = static_obs['position']
                distance = np.linalg.norm([new_x - sx, new_y - sy])
                if distance < (obs['size'] + static_obs['size']) / 2:
                    dx, dy = -dx, -dy

            # Cập nhật vị trí
            obs['position'] = [new_x, new_y]
            obs['velocity'] = [dx, dy]

        # Kiểm tra trạng thái va chạm
        done = self._check_collision()
        reward = self._calculate_reward(done)

        return self._get_state(), reward, done

    def _check_collision(self):
        """Kiểm tra va chạm giữa robot và chướng ngại vật."""
        rx, ry = self.robot.get_position()
        for obs in self.dynamic_obstacles:
            ox, oy = obs['position']
            distance = np.linalg.norm([rx - ox, ry - oy])
            if distance < (self.robot.size + obs['size']) / 2:
                return True
        for obs in self.static_obstacles:
            ox, oy = obs['position']
            distance = np.linalg.norm([rx - ox, ry - oy])
            if distance < (self.robot.size + obs['size']) / 2:
                return True
        return False

    def _calculate_reward(self, collision):
        """Tính reward dựa trên trạng thái."""
        if collision:
            return -100  # Phạt nếu va chạm
        goal_distance = np.linalg.norm([self.robot.x - self.goal[0], self.robot.y - self.goal[1]])
        return -goal_distance  # Phần thưởng âm dựa trên khoảng cách đến mục tiêu

    def _get_state(self):
        """Lấy trạng thái hiện tại."""
        state = {
            'robot': self.robot.get_position(),
            'dynamic_obstacles': [(obs['position'], obs['velocity']) for obs in self.dynamic_obstacles],
            'static_obstacles': [obs['position'] for obs in self.static_obstacles],
            'goal': self.goal
        }
        return state
