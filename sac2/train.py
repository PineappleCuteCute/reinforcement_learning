# 

import numpy as np
import pygame
from sac import SAC
from environment import RobotEnv  # Chắc chắn rằng bạn đã cài đặt class RobotEnv trong environment.py
import torch

# Khởi tạo môi trường và SAC
env = RobotEnv(width=800, height=600)  # Tạo môi trường với các tham số phù hợp
sac = SAC(state_size=2, action_size=4)  # Khởi tạo agent SAC

def train_sac(env, sac, episodes=1000):
    for episode in range(episodes):
        state = env.reset()  # Khởi tạo trạng thái ban đầu từ môi trường
        done = False
        total_reward = 0
        path = []  # Lưu trữ đường đi của robot từ điểm bắt đầu đến đích
        
        while not done:
            action = sac.select_action(state)  # SAC agent chọn hành động
            next_state, reward, done, _ = env.step(action)  # Thực hiện hành động
            sac.store_experience(state, action, reward, next_state, done)  # Lưu trữ kinh nghiệm
            sac.optimize()  # Tối ưu SAC
            state = next_state  # Cập nhật trạng thái
            total_reward += reward
            
            # Lưu trữ tọa độ robot để vẽ đường đi
            path.append(env.robot.position)
        
        # In ra kết quả của mỗi episode
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        
        # Hiển thị môi trường và vẽ đường đi nếu là các episode đặc biệt
        if episode % 100 == 0:
            env.render(path)  # Truyền đường đi để vẽ

# Chạy quá trình huấn luyện
train_sac(env, sac)

#save


