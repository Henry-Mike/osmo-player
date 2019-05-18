"""为模型提供训练环境。
采用了类似于 OpenAI gym 的 API 设计
"""

import math
from matplotlib import pyplot as plt
import numpy as np
import queue
import random
import sys

# add osmo.sessdsa to search path
sys.path.append('../osmo.sessdsa/src')

from consts import Consts
from world import World


plt.ion()


class PlayerProxy(object):
    """为 player 创建一个代理层。
    这是因为 Player.strategy 在 World.update 方法中写死了，导致无法从外部获取 action，
    player 无法与 OsmoEnv 交互。
    """
    def __init__(self, player):
        self.player = player
        self.action_queue = queue.Queue(maxsize=1)

    def __getattr__(self, name):
        return getattr(self.player, name)

    def reset(self):
        self.action_queue = queue.Queue(maxsize=1)

    def take_action(self, cells):
        return self.player.strategy(cells)

    def set_action(self, action):
        self.action_queue.put_nowait(action)

    def strategy(self, cells):
        """这个函数将作为代理函数被 World 调用"""
        return self.action_queue.get_nowait()


class OsmoEnv(object):
    """osmo 训练环境"""
    def __init__(self, player0, player1):
        player0 = PlayerProxy(player0)
        self.world = World(player0, player1)
        self.episode = -1
        self.total = -1


    def reset(self):
        self.world.new_game()
        self.world.player0.reset()
        self.episode += 1
        self.total += 1
        self.nframe = -1
        self.last_score = self.world.cells[self.world.player0.id].radius ** 2
        return self.world.cells


    def render(self):
        cells = self.world.cells
        image = get_artificial_image(cells)
        plt.imshow(image, interpolation='nearest')
        plt.title('episode - {}, frame - {}, total - {}'.format(self.episode, self.nframe, self.total))
        plt.pause(0.001)


    def step(self, action):
        """输入一个 action，返回下一个观测、奖励、以及是否结束"""

        self.total += 1
        self.nframe += 1

        self.world.player0.set_action(action)
        self.world.update(Consts["FRAME_DELTA"])

        me = self.world.cells[self.world.player0.id]
        enemy = self.world.cells[self.world.player1.id]
        if me.dead:
            return 'LOSE', -1, True
        if enemy.dead:
            return 'WIN', 1, True

        observation = self.world.cells
        cur_score = me.radius ** 2
        reward = (cur_score - self.last_score) / 1e4
        done = False
        self.last_score = cur_score

        return observation, reward, done


    def take_action(self):
        return self.world.player0.take_action(self.world.cells)


    def hold_on(self):
        input('press <enter> to exit ...')

    def close(self):
        pass


ZOOM = 4
MINI_WORLD_X = round(Consts['WORLD_X'] / ZOOM)
MINI_WORLD_Y = round(Consts['WORLD_Y'] / ZOOM)


def get_covered_square(cell):
    """获取一个 cell 覆盖的矩形区域（取圆的外接矩形）。
    覆盖区域可以有多种选择：圆形、外接矩形、十字线、竖线（代表直径）、横线、斜线等。
    """

    x, y = cell.pos
    r = cell.radius
    x, y, r = x / ZOOM, y / ZOOM, r / ZOOM
    xmin, xmax = x - r - 1, x + r + 1
    ymin, ymax = y - r - 1, y + r + 1
    xmin_int, xmax_int = math.floor(xmin), math.ceil(xmax)
    ymin_int, ymax_int = math.floor(ymin), math.ceil(ymax)

    i = np.arange(ymin_int, ymax_int + 1)
    j = np.arange(xmin_int, xmax_int + 1)
    area = np.ones((len(i), len(j)), dtype=np.float32)

    # 边界上未完全覆盖的格子
    area[0, :] *= ymin_int - ymin + 1
    area[-1, :] *= ymax - ymax_int + 1
    area[:, 0] *= xmin_int - xmin + 1
    area[:, -1] *= xmax - xmax_int + 1

    # 负的 index 不必处理，因为 Python 的切片默认支持负索引
    # 但越界的 index 需要处理
    j[j >= MINI_WORLD_X] -= MINI_WORLD_X
    i[i >= MINI_WORLD_Y] -= MINI_WORLD_Y

    return i, j, area


def get_covered_circle(cell):
    """获取一个 cell 覆盖的圆形区域"""

    x, y = cell.pos
    r = cell.radius
    x, y, r = x / ZOOM, y / ZOOM, r / ZOOM
    xmin, xmax = x - r - 1, x + r + 1
    ymin, ymax = y - r - 1, y + r + 1
    xmin_int, xmax_int = math.floor(xmin), math.ceil(xmax)
    ymin_int, ymax_int = math.floor(ymin), math.ceil(ymax)

    i = np.arange(ymin_int, ymax_int + 1)
    j = np.arange(xmin_int, xmax_int + 1)
    area = np.zeros((len(i), len(j)), dtype=np.float32)

    ii, jj = np.meshgrid(i, j, indexing='ij')
    distance = np.sqrt((jj - x) ** 2 + (ii - y) ** 2)
    area[distance <= r] = 1
    mask = (distance > r) & (distance <= r + 1)
    area[mask] = (r + 1 - distance)[mask]

    # 负的 index 不必处理，因为 Python 的切片默认支持负索引
    # 但越界的 index 需要处理
    j[j >= MINI_WORLD_X] -= MINI_WORLD_X
    i[i >= MINI_WORLD_Y] -= MINI_WORLD_Y

    return i, j, area


# 默认使用圆形覆盖
get_covered_area = get_covered_circle


def get_artificial_image(cells):
    """根据所有 cell 的覆盖情况，生成一个三通道图像，每个像素的值为 0~1。
    三个通道的值分别代表：我方 cell 覆盖、对方 cell 覆盖、其他 cell 覆盖。
    """
    image = np.zeros((MINI_WORLD_Y, MINI_WORLD_X, 3), dtype=np.float32)
    for i, cell in enumerate(cells):
        if cell.dead:
            continue
        channel = 0 if i == 0 else 1 if i == 1 else 2
        i, j, area = get_covered_area(cell)
        ii, jj = np.ix_(i, j)
        image[ii, jj, channel] += area
    image[image > 1] = 1.0
    return image
