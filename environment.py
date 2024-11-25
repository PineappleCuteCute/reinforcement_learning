# File: environment.py

import numpy as np
import random
from robot import Robot

class Environment:
    def __init__(self, width, height, num_obstacles=5):
        self.width = width
        self.height = height
        self.robot = Robot(width // 2, height // 2)  # Initialize robot at center
        self.obstacles = self.create_obstacles(num_obstacles)

    def create_obstacles(self, num):
        obstacles = []
        for _ in range(num):
            size = random.randint(10, 50)
            x = random.randint(size, self.width - size)
            y = random.randint(size, self.height - size)
            velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]
            obstacles.append({
                "position": [x, y],
                "size": size,
                "velocity": velocity
            })
        return obstacles

    def is_collision(self, position, size):
        for obs in self.obstacles:
            obs_pos = np.array(obs["position"])
            distance = np.linalg.norm(np.array(position) - obs_pos)
            if distance <= (size + obs["size"]) / 2:
                return True
        return False

    def update_obstacles(self):
        for obs in self.obstacles:
            for i in range(2):  # Update x and y
                obs["position"][i] += obs["velocity"][i]
                if obs["position"][i] <= 0 or obs["position"][i] >= self.width:
                    obs["velocity"][i] *= -1  # Reverse direction

    def step(self, action):
        """Perform one simulation step."""
        self.robot.perform_action(action)
        self.update_obstacles()

        # Check for collisions
        if self.is_collision(self.robot.position, self.robot.size):
            reward = -100  # Large penalty for collision
            done = True
        else:
            reward = -1  # Small penalty for each step
            done = False

        return self.robot.get_state(), reward, done

    def render(self):
        """Visualize environment (print obstacles and robot)."""
        print(f"Robot: Position={self.robot.position}, Velocity={self.robot.velocity}")
        for i, obs in enumerate(self.obstacles):
            print(f"Obstacle {i+1}: Position={obs['position']}, Size={obs['size']}, Velocity={obs['velocity']}")
