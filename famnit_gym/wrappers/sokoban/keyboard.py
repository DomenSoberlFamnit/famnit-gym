from typing import Optional

import gymnasium as gym
from gymnasium.core import ActType, ObsType
import pygame

import famnit_gym.envs.sokoban as sokoban

class Keyboard(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not sokoban.SokobanEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the SokobanEnv class.')

        self._env = env.unwrapped
        self._env._render_mode = 'human'
        
    def step(self, action: int):
        quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True

        action = -1
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            action = 0
        if keys[pygame.K_RIGHT]:
            action = 1
        if keys[pygame.K_DOWN]:
            action = 2
        if keys[pygame.K_LEFT]:
            action = 3

        if keys[pygame.K_SPACE]:
            self.env.reset()
        if keys[pygame.K_ESCAPE]:
            quit = True

        if quit:
            observation, _, _, _, info = self.env.step(action)
            return observation, 0, False, True, info

        if action == -1:
            self._env._update_frame()
            self._env._clock.tick(self._env.metadata['render_fps'])

        return self.env.step(action)