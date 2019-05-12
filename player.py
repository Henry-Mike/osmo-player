from collections import deque, namedtuple
import math
import random


Frame = namedtuple('Frame', ['index', 'cells', 'image'])


class Player(object):
    def __init__(self):
        self.count = 0
        self.last_frames = deque(maxlen=30)

    def play(self, cells):
        image = get_artificial_image(cells)
        frame = Frame(self.count, cells, image)

        self.last_frames.append(frame)
        self.count += 1


class BravePlayer(object):
    """总是冲向对手的勇敢 player"""
    def __init__(self, target=0, pmove=0.5):
        assert target in (0, 1)
        self.target_index = target
        self.me_index = 0 if target == 1 else 1
        self.pmove = pmove

    def play(self, cells):
        if random.random() > self.pmove:
            return None
        me = cells[self.me_index]
        target = cells[self.target_index]
        theta = math.atan2(me.pos[0] - target.pos[0], me.pos[1] - target.pos[1])
        return theta


class MadPlayer(object):
    """总是像一个方向冲的 player"""
    def __init__(self, theta=0):
        self.theta = theta

    def play(self, cells):
        return self.theta


class AimlessPlayer(object):
    """不停地向任意方向发射的 player"""
    def __init__(self):
        pass

    def play(self, cells):
        return random.random() * 2 * math.pi


class NormalPlayer(object):
    """一个比较正常且通用的 player"""
    def __init__(self, pmove=0.2):
        self.pmove = pmove

    def play(self, cells):
        if random.random() < self.pmove:
            return random.random() * 2 * math.pi
