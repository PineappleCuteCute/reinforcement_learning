# environment.py

import gym
import numpy as np
import matplotlib.pyplot as plt

class RobotEnv(gym.Env):
    def __init__(self, grid_size=10, goal=(9, 9), obstacles=None):
        self.grid_size = grid_size
        self.goal = goal
        self.robot_position = (0, 0)  # Vị trí xuất phát
        self.obstacles = obstacles or [(3, 3), (4, 4), (5, 5)]  # Chướng ngại vật mặc định
        self.action_space = gym.spaces.Discrete(4)  # 4 hành động: lên, xuống, trái, phải
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
    def reset(self):
        self.robot_position = (0, 0)
        return np.array(self.robot_position)
    
    def step(self, action):
        x, y = self.robot_position
        
        if action == 0:  # Di chuyển lên
            x = max(0, x - 1)
        elif action == 1:  # Di chuyển xuống
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Di chuyển trái
            y = max(0, y - 1)
        elif action == 3:  # Di chuyển phải
            y = min(self.grid_size - 1, y + 1)
        
        new_position = (x, y)
        
        # Kiểm tra chướng ngại vật
        if new_position in self.obstacles:
            return np.array(new_position), -10, False, {}  # Phạt lớn khi va phải chướng ngại vật
        
        self.robot_position = new_position
        
        # Nếu đến đích
        if self.robot_position == self.goal:
            return np.array(self.robot_position), 100, True, {}
        
        # Nếu chưa đến đích
        return np.array(self.robot_position), -1, False, {}
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal] = 2  # Mục tiêu
        for obs in self.obstacles:
            grid[obs] = 1  # Chướng ngại vật
        grid[self.robot_position] = 3  # Robot
        
        plt.imshow(grid, cmap="hot", interpolation="nearest")
        plt.show()
