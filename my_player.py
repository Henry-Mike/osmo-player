from collections import deque, namedtuple
import functools
import math
import os
import random

import numpy as np

from consts import Consts

from env import get_nearest_n_cells, xy_to_theta
from ddpg import create_agent, nb_actions, observation_shape

cur_dir = os.path.dirname(os.path.abspath(__file__))
weights_file = os.path.join(cur_dir, 'weights.h5')

agent = create_agent(nb_actions, observation_shape)
agent.load_weights(weights_file)


class MyPlayer():
    def __init__(self, id, **kwargs):
        self.id = id
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(name)

    def take_action(self, observation):
        action = agent.forward(observation)
        return action

    def strategy(self, cells):
        observation = get_nearest_n_cells(cells, id=self.id)
        action = agent.forward(observation)
        theta = xy_to_theta(action)
        return theta
