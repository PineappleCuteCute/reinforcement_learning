import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

# Thiết lập các tham số thông qua argparse
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
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

# Cấu trúc lưu log_prob và value
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Định nghĩa mô hình Actor-Critic
class Policy(nn.Module):
    """
    Triển khai cả actor và critic trong cùng một mô hình
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)  # Lớp fully-connected đầu vào với 128 đơn vị

        # Lớp của actor
        self.action_head = nn.Linear(128, 2)

        # Lớp của critic
        self.value_head = nn.Linear(128, 1)

        # Bộ nhớ cho hành động và phần thưởng
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        Truyền qua của cả actor và critic
        """
        x = F.relu(self.affine1(x))  # Kích hoạt bằng ReLU

        # Actor: chọn hành động từ trạng thái s_t bằng cách trả về xác suất của mỗi hành động
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # Critic: đánh giá trạng thái s_t
        state_values = self.value_head(x)

        # Trả về giá trị cho cả actor và critic (tuple gồm xác suất và giá trị)
        return action_prob, state_values

# Khởi tạo mô hình và bộ tối ưu
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()  # Nhỏ nhất để tránh chia cho 0

# Hàm để chọn hành động từ trạng thái hiện tại
def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # Tạo phân phối xác suất của các hành động
    m = Categorical(probs)

    # Lấy mẫu hành động từ phân phối
    action = m.sample()

    # Lưu vào bộ nhớ hành động
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # Trả về hành động để thực hiện (trái hoặc phải)
    return action.item()

# Hàm kết thúc một episode và thực hiện cập nhật gradient
def finish_episode():
    """
    Huấn luyện: Tính toán actor và critic loss và thực hiện backpropagation
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # Lưu loss của actor
    value_losses = []  # Lưu loss của critic
    returns = []  # Lưu giá trị thực

    # Tính giá trị thực từ phần thưởng trả về từ môi trường
    for r in model.rewards[::-1]:
        R = r + args.gamma * R  # Giá trị giảm dần
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # Tính actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # Tính critic (value) loss sử dụng L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # Reset gradient
    optimizer.zero_grad()

    # Tổng hợp các giá trị của policy_losses và value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # Thực hiện backpropagation
    loss.backward()
    optimizer.step()

    # Xóa bộ nhớ phần thưởng và hành động
    del model.rewards[:]
    del model.saved_actions[:]

# Hàm chính để huấn luyện tác nhân
def main():
    running_reward = 10

    # Vòng lặp cho các episode
    for i_episode in count(1):

        # Reset môi trường và phần thưởng của episode
        state, _ = env.reset()
        ep_reward = 0

        # Mỗi episode chỉ chạy 9999 bước để tránh vòng lặp vô hạn
        for t in range(1, 10000):
            # Chọn hành động từ chính sách
            action = select_action(state)

            # Thực hiện hành động
            state, reward, done, _, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # Cập nhật phần thưởng tích lũy
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Thực hiện backpropagation
        finish_episode()

        # Ghi nhận kết quả
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # Kiểm tra nếu bài toán đã được "giải"
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

# Chạy hàm chính khi script được thực thi
if __name__ == '__main__':
    main()
