import math
from matplotlib import pyplot as plt
import numpy as np

from consts import Consts


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

    # 索引跨边界时，边界上的 cover 需要复制一份
    xoverflow = j < 0
    if any(xoverflow):
        insert_to = np.where(j == 0)[0][0]
        j[xoverflow] -= 1
        j = np.insert(j, insert_to, -1)
        area = np.insert(area, insert_to, area[:, insert_to], axis=1)

    yoverflow = i < 0
    if any(yoverflow):
        insert_to = np.where(i == 0)[0][0]
        i[yoverflow] -= 1
        i = np.insert(i, insert_to, -1)
        area = np.insert(area, insert_to, area[insert_to], axis=0)

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

    # 索引跨边界时，边界上的 cover 需要复制一份
    xoverflow = j < 0
    if any(xoverflow):
        insert_to = np.where(j == 0)[0][0]
        j[xoverflow] -= 1
        j = np.insert(j, insert_to, -1)
        area = np.insert(area, insert_to, area[:, insert_to], axis=1)

    yoverflow = i < 0
    if any(yoverflow):
        insert_to = np.where(i == 0)[0][0]
        i[yoverflow] -= 1
        i = np.insert(i, insert_to, -1)
        area = np.insert(area, insert_to, area[insert_to], axis=0)

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


class Player():
    def __init__(self):
        pass

    def play(self, cells):
        image = get_artificial_image(cells)
        plt.imshow(image, interpolation='nearest')
        plt.show()
