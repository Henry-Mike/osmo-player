"""训练场"""

from env import OsmoEnv
from my_player import MyPlayer
import time


me = MyPlayer(0, 'normal', pmove=0.2)
enemy = MyPlayer(1, 'brave', target=0, pmove=0.3)
# enemy = MyPlayer(1, 'mad', theta=0)
# enemy = MyPlayer(1, 'aimless')

env = OsmoEnv(me, enemy)

observation = env.reset()
for i in range(100):
    env.render()

    # 请使用你的 player 产生 action
    action = env.take_action()
    observation, reward, done = env.step(action)

    action_str = 'None' if action is None else '{:6.4f}'.format(action)
    print('round {:03d}, action: {:>6s}, reward: {:9.6f}, done: {}'.format(i, action_str, reward, done))

    if done:
        observation = env.reset()
    time.sleep(0.005)

env.hold_on()
env.close()
