"""训练模型"""

from env import OsmoEnv
from player import Player
import time


env = OsmoEnv()
player = Player()


observation = env.reset()
for i in range(100):
    env.render()

    # 请使用你的 player 产生 action
    action = env.random_action()
    print('round {}, action: {}'.format(i, action))

    observation, reward, done = env.step(action)
    if done:
        observation = env.reset()
    time.sleep(0.02)

env.close()
