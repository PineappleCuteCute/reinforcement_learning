import torch
import numpy as np
from sac import SAC
from robot import Robot
from environment import Environment

def train_sac():
    # Khởi tạo robot, môi trường và SAC
    start_pos = [0, 0]  # Vị trí ban đầu của robot
    goal_pos = [5, 5]  # Vị trí đích
    obstacles = [[1, 2], [3, 3], [4, 1]]  # Chướng ngại vật tĩnh

    robot = Robot(start_pos)
    env = Environment(goal_pos, obstacles)

    state_size = 2  # Robot di chuyển trong không gian 2D
    action_size = 4  # 4 hướng di chuyển (lên, xuống, trái, phải)
    sac = SAC(state_size, action_size)

    # Huấn luyện SAC
    num_episodes = 500
    for episode in range(num_episodes):
        state = env.reset(robot)
        done = False
        total_reward = 0
        while not done:
            action = sac.select_action(state)  # Chọn hành động từ SAC
            next_state, reward, done = env.step(action, robot)  # Tiến hành hành động trong môi trường
            sac.store_experience(state, action, reward, next_state, done)  # Lưu trữ kinh nghiệm
            sac.optimize()  # Tối ưu hóa SAC
            state = next_state
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}")
    
    print("Training Finished!")

if __name__ == "__main__":
    train_sac()
