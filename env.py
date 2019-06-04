"""为模型提供训练环境。
采用了类似于 OpenAI gym 的 API 设计
"""

import copy
import functools
import math
from matplotlib import pyplot as plt
import numpy as np
import queue
import random
import sys

# add osmo.sessdsa to search path
sys.path.append('../osmo.sessdsa/src')

from consts import Consts
from world import World as OriginWorld


OBSERVE_N = 5
OBSERVE_M = 6
MIN_RADIUS = 3
if MIN_RADIUS is None:
    MIN_RADIUS = Consts['DEFAULT_RADIUS'] * Consts['EJECT_MASS_RATIO'] ** 0.5

plt.ion()


class World(OriginWorld):
    """对战平台的 World 类不合用，打个补丁"""

    def check_point(self, lose0, lose1, cause):
        """
        Args:
            lose0: mark the status of player0.
            lose1: mark the status of player1.
            cause: reason for the end of the game.
        Returns:
            who is the winner.
        """
        if not lose0 and lose1:
            self.game_over(0, cause, (lose0, lose1))
            self.winner = 0
        elif lose0 and not lose1:
            self.game_over(1, cause, (lose0, lose1))
            self.winner = 1
        elif lose0 and lose1:
            self.game_over(-1, cause, (lose0, lose1))
            self.winner = -1
        else:
            self.winner = None
        return bool(lose0 or lose1)


class RandomPlayer():
    def __init__(self, id, strategy='normal', **kwargs):
        self.id = id
        self.kwargs = kwargs

        strategy_fun = getattr(self, strategy + '_strategy')
        if strategy_fun is None:
            raise ValueError('no strategy "{}"'.format(strategy))
        setattr(self, 'strategy', strategy_fun)

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(name)

    @property
    def enemy_id(self):
        return 0 if self.id == 1 else 1

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


class Space():
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.random.random(shape)


def xy_to_theta(xy, threshold=0.3):
    """把直角坐标系下的 x,y 转化为极坐标下的 theta（范围 0~2*pi）。
    如果 x, y 向量的长度小于 threshold，则返回 None
    """
    x, y = xy
    c = complex(x, y)
    if abs(c) < threshold:
        return None

    theta = np.angle(c)
    if theta < 0:
        theta += 2 * math.pi
    return theta


class OsmoEnv(object):
    """osmo 训练环境"""
    def __init__(self):
        self.player0 = PlayerProxy(RandomPlayer(0, 'normal', pmove=0.2))
        self.player1 = RandomPlayer(1, 'brave', target=0, pmove=0.3)
        self.world = World(self.player0, self.player1)

        self.observation_space = Space(shape=(OBSERVE_N, OBSERVE_M))
        # 输出两个值 vx, vy，可从中推导出发射角度（为什么不直接输出一个角度呢？
        # vx、vy 是单调连续的，更容易拟合；如果是一个角度，0度和360度在数值上位于激活函数的两端，
        # 但实际上二者代表非常相近的角度，因此不容易拟合）
        self.action_space = Space(shape=(2,))

        self.get_observation = functools.partial(get_nearest_n_cells, id=self.player0.id)

        self.episode = -1
        self.total = -1
        self.fig = None


    def reset(self):
        self.world.new_game()
        self.player0.reset()
        self.episode += 1
        self.total += 1
        self.nframe = -1
        self.last_score = self.world.cells[0].radius ** 2
        return self.get_observation(copy.deepcopy(self.world.cells))[0]


    def render(self, mode=None):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(211)
            self.ax1 = self.fig.add_subplot(212)
            self.cbar = None
        observation, marked = self.get_observation(copy.deepcopy(self.world.cells))
        image = get_image(self.world.cells, marked)
        cax = self.ax1.matshow(observation.transpose())
        self.ax.imshow(image, interpolation='nearest')
        if not self.cbar:
            self.cbar = self.fig.colorbar(cax, orientation='horizontal')
        self.fig.suptitle('episode - {}, frame - {}, total - {}'.format(self.episode, self.nframe, self.total))
        plt.pause(0.001)


    def step(self, action):
        """输入一个 action，返回下一个观测、奖励、以及是否结束"""

        self.total += 1
        self.nframe += 1

        theta = xy_to_theta(action)
        self.player0.set_action(theta)
        self.world.update(Consts["FRAME_DELTA"])

        observation = self.get_observation(copy.deepcopy(self.world.cells))[0]
        done = False
        cur_score = self.world.cells[0].radius ** 2
        reward = cur_score - self.last_score

        # 减少对发射炮弹的惩罚
        if reward < 0:
            reward *= 0.6

        self.last_score = cur_score
        info = {}

        if self.world.winner == 0:
            x0, y0 = self.world.cells[0].pos
            r0 = self.world.cells[0].radius
            x1, y1 = self.world.cells[1].pos
            r1 = self.world.cells[1].radius
            if abs(complex(x0 - x1, y0 - y1)) < r0 + r1:
                reward = 1000
            done = True
        elif self.world.winner == 1:
            reward = -1000
            done = True
        else:
            if self.world.winner == -1:
                done = True
            # 每生存下来一步，给予一定奖励
            reward += 0.5

        # reward /= 1000
        info['reward'] = reward
        return observation, reward, done, info


    def take_action(self):
        return self.player0.take_action(copy.deepcopy(self.world.cells))


    def hold_on(self):
        input('press <enter> to exit ...')

    def close(self):
        pass


def get_nearest_n_cells(cells, id=0):
    """取距离最近的 n 个 cell 作为网络的输入"""

    # n 至少为 2
    n = max(2, OBSERVE_N)

    # 默认将我方 cell 排在第一位
    if id != 0:
        cells[0], cells[1] = cells[1], cells[0]

    # 将所有 cell 的特征提取出来，组成一个二维矩阵，每一行代表一个 cell
    # 最后追加一列，用来标识当前 cell 是否为敌方 cell
    mat = [
        [*cell.pos, cell.radius, *cell.veloc, 1 if cell.id < 2 and cell.id != id else 0, cell.id]
        for cell in cells if not cell.dead]
    mat = np.array(mat)

    # # 最后追加一个长度为 3 的 one-hot 向量，用来标识 我方、敌方、普通 cell
    # mapper = {0: [1, 0, 0], 1: [0, 1, 0]}
    # indicator = np.array([mapper.get(i, [0, 0, 1]) for i in range(len(cells))])
    # mat = np.concatenate([mat, indicator], axis=1)

    # 做坐标系转换，将我方 cell 平移到场地正中央，这样求距离时就避免了跨界的问题
    cell0 = cells[id]
    x0, y0 = cell0.pos
    delta_x = Consts['WORLD_X'] / 2 - x0
    delta_y = Consts['WORLD_Y'] / 2 - y0

    xs = mat[:, 0]
    ys = mat[:, 1]
    xs += delta_x
    ys += delta_y
    xs[xs < 0] += Consts['WORLD_X']
    ys[ys < 0] += Consts['WORLD_Y']
    xs[xs > Consts['WORLD_X']] -= Consts['WORLD_X']
    ys[ys > Consts['WORLD_Y']] -= Consts['WORLD_Y']

    # 求其他 cell 与我方 cell 的 diff
    xs -= xs[0]
    ys -= ys[0]

    # # 将除敌我双方外的 cell 按照 距离、大小、x 坐标、y 坐标 排序。半径过小的 cell 直接排在末尾。
    # 将所有 cell 按照 距离、大小、x 坐标、y 坐标 排序。半径过小的 cell 直接排在末尾。
    arr_for_sort = np.array(
        [(abs(complex(row[0], row[1])) if row[2] > MIN_RADIUS else np.inf, -row[2], row[0], row[1]) for row in mat],
        dtype=[('dist', 'f8'), ('dr', 'f8'), ('dx', 'f8'), ('dy', 'f8')])
    index = arr_for_sort.argsort()
    mat = mat[index]


    # 归一化，所有参数到 -1~1
    # dx, dy 划归到 -1~1
    mat[:, 0] = mat[:, 0] / (Consts['WORLD_X'] / 2)
    mat[:, 1] = mat[:, 1] / (Consts['WORLD_Y'] / 2)

    # 半径折算为对我方 cell 半径的倍数
    mat[:, 2] = mat[:, 2] / mat[0, 2] - 1
    # mat[:, 2] = mat[:, 2] / abs(mat[:, 2]).max()

    # vx, vy 划归到 -1~1
    # 归一化之后的值太小，需要放大一些
    mat[:, 3:5] = (mat[:, 3:5] - mat[0, 3:5]) / Consts['MAX_VELOC'] * 15

    # 观测对象中不包含自身
    offset = 1
    n += offset

    # 取出前 n 个 cell，如果不足 n 个，以 0 填充。
    if mat.shape[0] < n:
        mat = np.concatenate([mat, [[0] * mat.shape[1]] * (n - mat.shape[0])], axis=0)\

    # 标记在视野范围内的 cells
    marked = set(mat[offset:n, -1].astype(int)) - {0, 1}

    return mat[offset:n, :OBSERVE_M], marked


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


def get_image(cells, marked=None):
    """根据所有 cell 的覆盖情况，生成一个三通道图像，每个像素的值为 0~1。
    三个通道的值分别代表：我方 cell 覆盖、对方 cell 覆盖、其他 cell 覆盖。
    """
    image = np.zeros((MINI_WORLD_Y, MINI_WORLD_X, 3), dtype=np.float32)
    for i, cell in enumerate(cells):
        if cell.dead:
            continue
        channel = [0] if i == 0 else [1] if i == 1 else [2]
        if marked is not None and cell.id in marked:
            channel = [1, 2]
        i, j, area = get_covered_area(cell)
        ii, jj = np.ix_(i, j)
        for c in channel:
            image[ii, jj, c] += area
    image[image > 1] = 1.0
    return image
