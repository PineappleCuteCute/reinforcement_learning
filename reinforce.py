import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Thiết lập các tham số thông qua argparse
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# Tạo môi trường CartPole và thiết lập seed ngẫu nhiên
env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

# Định nghĩa mạng chính sách
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)  # Lớp fully-connected đầu tiên với 128 đơn vị
        self.dropout = nn.Dropout(p=0.6)  # Dropout với tỷ lệ 0.6
        self.affine2 = nn.Linear(128, 2)  # Lớp fully-connected thứ hai cho hai hành động

        self.saved_log_probs = []  # Danh sách để lưu log của xác suất các hành động đã chọn
        self.rewards = []  # Danh sách để lưu phần thưởng của từng bước

    def forward(self, x):
        x = self.affine1(x)  # Truyền qua lớp affine1
        x = self.dropout(x)  # Áp dụng dropout
        x = F.relu(x)  # Áp dụng hàm kích hoạt ReLU
        action_scores = self.affine2(x)  # Truyền qua lớp affine2
        return F.softmax(action_scores, dim=1)  # Áp dụng hàm softmax để tính xác suất

# Khởi tạo chính sách và bộ tối ưu
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()  # Nhỏ nhất để tránh chia cho 0

# Hàm để chọn hành động dựa trên trạng thái hiện tại
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

# Hàm kết thúc một episode và thực hiện cập nhật gradient
def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    # Tính giá trị hồi quy cho từng bước
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    # Chuẩn hóa giá trị hồi quy để ổn định huấn luyện
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # Tính toán policy loss dựa trên log của xác suất và giá trị hồi quy
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    # Xóa lịch sử của phần thưởng và xác suất đã chọn
    del policy.rewards[:]
    del policy.saved_log_probs[:]

# Hàm chính để huấn luyện tác nhân
def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

# Chạy hàm chính khi script được thực thi
if __name__ == '__main__':
    main()
