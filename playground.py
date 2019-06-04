"""训练场"""

import math
import time

import numpy as np

from env import OsmoEnv, xy_to_theta
from my_player import MyPlayer


player = MyPlayer(0)
env = OsmoEnv()

observation = env.reset()
for i in range(300):
    env.render()

    action = player.take_action(observation)
    observation, reward, done, info = env.step(action)

    theta = xy_to_theta(action)
    theta = 'None' if theta is None else '{:6.4f}'.format(theta)
    print('round {:03d}, action: {:>6s}, reward: {:7.2f}, done: {}'.format(i, theta, reward, done))

    if done:
        observation = env.reset()
    # time.sleep(0.001)

env.hold_on()
env.close()
