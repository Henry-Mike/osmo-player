import numpy as np
import random

random.seed(100)
np.random.seed(100)

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from env import OsmoEnv

env = OsmoEnv()

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
observation_shape = env.observation_space.shape


def create_agent(nb_actions, observation_shape):
    """构造 ddpg agent"""

    import os
    import sys
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    keras_rl = os.path.join(os.path.dirname(cur_dir), 'keras-rl')
    sys.path.insert(0, keras_rl)

    from rl.agents import DDPGAgent
    from rl.memory import SequentialMemory
    from rl.random import OrnsteinUhlenbeckProcess

    # 构造 Actor
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    print(actor.summary())

    # 构造 critic
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + observation_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # 编译模型
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.6, mu=0, sigma=0.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10, batch_size=64,
                      random_process=random_process, gamma=.999, target_model_update=1e-3)
    agent.compile([Adam(lr=.001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)], metrics=['mae'])

    return agent


if __name__ == '__main__':
    agent = create_agent(nb_actions, observation_shape)

    # 训练模型
    agent.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=None)

    # 保存权重
    agent.save_weights('weights.h5', overwrite=True)

    # 测试模型
    agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=200)
