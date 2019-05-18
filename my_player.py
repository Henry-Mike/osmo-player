from collections import deque, namedtuple
import math
import random

from player import Player

Frame = namedtuple('Frame', ['index', 'cells', 'image'])


class MyPlayer(Player):
    def __init__(self, id, strategy='normal', **kwargs):
        super().__init__(id)
        self.kwargs = kwargs
        strategy_fun = getattr(self, strategy + '_strategy')
        if strategy_fun is None:
            raise ValueError('no strategy "{}"'.format(strategy))
        setattr(self, 'strategy', strategy_fun)
        # self.count = 0
        # self.last_frames = deque(maxlen=30)

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(name)

    @property
    def enemy_id(self):
        return 0 if self.id == 1 else 1

    def my_strategy(self, cells):
        image = get_artificial_image(cells)
        frame = Frame(self.count, cells, image)

        self.last_frames.append(frame)
        self.count += 1

    def brave_strategy(self, cells):
        """总是冲向对手"""
        if random.random() > self.pmove:
            return None
        me = cells[self.id]
        enemy = cells[self.enemy_id]
        theta = math.atan2(me.pos[0] - enemy.pos[0], me.pos[1] - enemy.pos[1])
        return theta

    def mad_strategy(self, cells):
        """不间断地向同一个方向发射"""
        return self.theta

    def aimless_strategy(self, cells):
        """不间断地向任意方向发射"""
        return random.random() * 2 * math.pi

    def normal_strategy(self, cells):
        """以一定概率向任意方向发射"""
        if random.random() < self.pmove:
            return random.random() * 2 * math.pi

