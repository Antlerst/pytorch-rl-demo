import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count


class PGN(nn.Module):
    def __init__(self):
        super(PGN, self).__init__()
        self.linear1 = nn.Linear(4, 24)
        self.linear2 = nn.Linear(24, 36)
        self.linear3 = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 超参数
    BATCH_SIZE = 5
    LEARNING_RATE = 0.01
    GAMMA = 0.99

    env = gym.make('CartPole-v1')
    pgn = PGN()
    optimizer = torch.optim.RMSprop(pgn.parameters(), lr=LEARNING_RATE)

    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    num_episodes = 500
    for i_episode in range(num_episodes):
        next_state = env.reset()
        env.render(mode='rgb_array')

        for t in count():
            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            probs = pgn(state)
            m = Bernoulli(probs)
            action = m.sample()

            action = action.data.numpy().astype(int).item()
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array')

            # 让终止action的reward为0
            if done:
                reward = 0

            # 缓存state、action、reward和步数
            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)
            steps += 1

            if done:
                logger.info({'Episode {}: durations {}'.format(i_episode, t)})
                break

        # 按照Batch Size更新代码
        if i_episode > 0 and i_episode % BATCH_SIZE == 0:
            # 根据某次action之后所有特定action的reward，提升当前reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * GAMMA + reward_pool[i]
                    reward_pool[i] = running_add

            # 均一化reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Policy Gradient
            optimizer.zero_grad()

            for i in range(steps):
                # 某次游戏一共进行了steps步
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = pgn(state)
                m = Bernoulli(probs)
                loss = -m.log_prob(action) * reward
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
