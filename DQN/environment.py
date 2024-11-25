import numpy as np
import random
from robot import Robot

class Environment:
    def __init__(self, width, height, num_dynamic_obs=5, num_static_obs=5):
        self.width = width
        self.height = height
        self.robot = Robot(x=width // 2, y=height // 2, size=20)
        self.dynamic_obstacles = self._create_dynamic_obstacles(num_dynamic_obs)
        self.static_obstacles = self._create_static_obstacles(num_static_obs)
        self.goal = [width - 40, height - 40]

    def _create_dynamic_obstacles(self, num):#định nghĩa chướng ngại vật động!!!!!
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(10, 20)
            velocity = [random.choice([-1, 1]), random.choice([-1, 1])]
            obstacles.append({'position': [x, y], 'size': size, 'velocity': velocity})
        return obstacles

    def _create_static_obstacles(self, num):#định nghĩa chướng ngại vật tĩnh!!!!!!
        obstacles = []
        for _ in range(num):
            x, y = random.randint(40, self.width - 40), random.randint(40, self.height - 40)
            size = random.randint(20, 50)
            obstacles.append({'position': [x, y], 'size': size})
        return obstacles

    def step(self, action): #Robot nhận action từ env.step() và di chuyển theo vector này
        self.robot.move(action)
        self._update_moving_obstacles()
        done = self._check_collision()
        reward = self._calculate_reward(done)
        return self._get_state(), reward, done

    def _update_moving_obstacles(self): #cập nhật trạng thái hành động cho cnv động
        for obs in self.dynamic_obstacles:
            dx, dy = obs['velocity']
            obs['position'][0] += dx * 5
            obs['position'][1] += dy * 5
            if obs['position'][0] < 0 or obs['position'][0] > self.width:
                obs['velocity'][0] = -dx
            if obs['position'][1] < 0 or obs['position'][1] > self.height:
                obs['velocity'][1] = -dy

    def _check_collision(self): #kiểm tra xem robot có va chạm cnv hay không -> reward!!!!!
        rx, ry = self.robot.get_position()
        for obs in self.dynamic_obstacles + self.static_obstacles:
            ox, oy = obs['position']
            distance = np.linalg.norm([rx - ox, ry - oy])
            if distance < (self.robot.size + obs['size']) / 2:
                return True
        return False

    def _calculate_reward(self, collision):#định nghĩa reward!!!!!!
        if collision: #Robot va chạm với chướng ngại vật: reward = -100.
            return -100
        goal_distance = np.linalg.norm([self.robot.x - self.goal[0], self.robot.y - self.goal[1]])
        return -goal_distance #Robot di chuyển gần hơn tới điểm đích: reward = -distance_to_goal

    def _get_state(self):#định nghĩa trạng thái!!!!!!
        #Vị trí và vận tốc của các chướng ngại vật động (x, y, dx, dy) cho mỗi chướng ngại vật
        """Trả về trạng thái hiện tại dưới dạng vector."""
        # Lấy vị trí robot
        state = list(self.robot.get_position())

        # Thêm vị trí và vận tốc của chướng ngại vật động
        for obs in self.dynamic_obstacles:
            state.extend(obs['position'])
            state.extend(obs['velocity'])

        # Thêm vị trí của chướng ngại vật tĩnh
        for obs in self.static_obstacles:
            state.extend(obs['position'])

        # Thêm vị trí mục tiêu
        state.extend(self.goal)

        return np.array(state, dtype=np.float32)  # Trả về mảng numpy


    def reset(self):
        self.robot = Robot(x=self.width // 2, y=self.height // 2, size=20)
        self.dynamic_obstacles = self._create_dynamic_obstacles(len(self.dynamic_obstacles))
        self.static_obstacles = self._create_static_obstacles(len(self.static_obstacles))
        return self._get_state()
