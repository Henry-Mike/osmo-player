#!/usr/bin/env python3

# This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.


import math
from matplotlib import pyplot as plt
import numpy as np


from consts import Consts
from cell import Cell


ZOOM = 4
MINI_WORLD_X = round(Consts['WORLD_X'] / ZOOM)
MINI_WORLD_Y = round(Consts['WORLD_Y'] / ZOOM)


def get_covered_square(cell):
    """获取一个 cell 覆盖的所有坐标。
    覆盖区域被简化为圆的外接矩形。
    为了使覆盖区域在边缘处也更加精确，采用了类似“消锯齿”的技术，对每个覆盖的格点赋予一个 0~1 之间的“覆盖程度”。
    覆盖区域可以有多种选择：圆形、外接矩形、十字线、竖线（代表直径）、横线、斜线等。
    """

    def get_cover_1d(start, end):
        istart = math.floor(start)
        iend = math.ceil(end)
        cover = np.arange(istart, iend + 1, dtype=np.int32)
        factor = np.ones(cover.shape, dtype=np.float32)
        factor[0] = istart - start + 1
        factor[-1] = end - iend + 1
        return cover, factor

    x, y = cell.pos
    r = cell.radius
    x, y, r = x / ZOOM, y / ZOOM, r / ZOOM

    xi, xf = get_cover_1d(x -r, x + r)
    yi, yf = get_cover_1d(y -r, y + r)

    # 负的 index 不必处理，因为 Python 的切片默认支持负索引
    # 但越界的 index 需要处理
    xi[xi >= MINI_WORLD_X] -= MINI_WORLD_X
    yi[yi >= MINI_WORLD_Y] -= MINI_WORLD_Y

    # 索引跨边界时，边界上的索引需要一分为二
    xmask = xi < 0
    if any(xmask):
        xi[xmask] -= 1
        insert_to = np.where(xi==0)[0][0]
        xi = np.insert(xi, insert_to, -1)
        xf = np.insert(xf, insert_to, 1)

    ymask = yi < 0
    if any(ymask):
        yi[ymask] -= 1
        insert_to = np.where(yi==0)[0][0]
        yi = np.insert(yi, insert_to, -1)
        yf = np.insert(yf, insert_to, 1)

    return xi, yi, xf, yf


def get_artificial_image(cells):
    """根据所有 cell 的覆盖情况，生成一个三通道图像，每个像素的值为 0~1。
    三个通道的值分别代表：我方 cell 覆盖、对方 cell 覆盖、其他 cell 覆盖。
    """
    image = np.zeros((MINI_WORLD_Y, MINI_WORLD_X, 3), dtype=np.float32)
    for i, cell in enumerate(cells):
        channel = 0 if i == 0 else 1 if i == 1 else 2
        xi, yi, xf, yf = get_covered_square(cell)
        yyi, xxi = np.ix_(yi, xi)
        image[yyi, xxi, channel] += np.dot(yf[:, np.newaxis], xf[np.newaxis, :])
    image[image > 1] = 1.0
    return image


class Player():
    def __init__(self):
        pass

    def play(self, cells):
        image = get_artificial_image(cells)
        plt.imshow(image, interpolation='nearest')
        plt.show()
