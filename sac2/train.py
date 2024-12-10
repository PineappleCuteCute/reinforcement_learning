# train.py

import numpy as np
from sac import SAC
from environment import RobotEnv

# state_size = 2  # Kích thước không gian trạng thái (ví dụ: 2)
# action_size = 4  # Kích thước không gian hành động (ví dụ: 4)

# # Khởi tạo SAC với các tham số đúng
# sac = SAC(state_size, action_size)


def train_sac(env, sac, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            sac.store_experience(state, action, reward, next_state, done)
            sac.optimize()
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        
        if episode % 100 == 0:
            env.render()

# Khởi tạo môi trường và SAC
env = RobotEnv()
sac = SAC(state_size=2, action_size=4)

train_sac(env, sac)
