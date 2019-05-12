"""训练场"""

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
    observation, reward, done = env.step(action)

    action_str = 'None' if action is None else '{:6.4f}'.format(action)
    print('round {:03d}, action: {:>6s}, reward: {:8.2f}, done: {}'.format(i, action_str, reward, done))

    if done:
        observation = env.reset()
    time.sleep(0.005)

env.hold_on()
env.close()
