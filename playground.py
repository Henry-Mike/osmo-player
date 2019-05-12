"""训练场"""

from env import OsmoEnv
from player import Player, BravePlayer, MadPlayer, AimlessPlayer, NormalPlayer
import time


env = OsmoEnv()
player = Player()
brave_enemy = BravePlayer(target=0, pmove=0.3)
mad_enemy = MadPlayer(theta=0)
aimless_enemy = AimlessPlayer()
normal_enemy = NormalPlayer(pmove=0.2)

env.set_enemy_player(brave_enemy)

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
