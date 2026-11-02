from typing import Optional
import math
import numpy as np
import gymnasium as gym
import famnit_gym
from gymnasium.core import ActType, ObsType
import pygame

import famnit_gym.envs.astronaut as astronaut

class Asteroids(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not astronaut.AstronautEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the AstronautEnv class.')        

        # Underlying methods.
        self._base_init_graphics = self.env.get_wrapper_attr("_init_graphics")
        self._base_simulation_step = self.env.get_wrapper_attr("_simulation_step")
        self._base_get_obs = self.env.get_wrapper_attr("_get_obs")
        self._base_get_info = self.env.get_wrapper_attr("_get_info")
        self._base_get_reward_info = self.env.get_wrapper_attr("_get_reward_info")

        # Graphics for this wrapper not yet initialized.
        self._graphics_initialized = False

    def _get_obs(self):
        obs = np.array(
            [
                self._asteroid_front['x'],
                self._asteroid_front['y']
            ],
            dtype=np.float32
        )
        
        return np.concatenate([self._base_get_obs(), obs])

    def _get_info(self):
        info = self._base_get_info()
        info['variables'] = info['variables'] + ['asteroid_x', 'asteroid_y']
        
        return info

    def _get_reward_info(self):
        info = self._get_info()
        reward_info = self._base_get_reward_info()
        reward_info['collision'] = False if info['collision'] == 'no' else True

        return reward_info

    def _init_graphics(self):
        global pygame
        import pygame

        self._base_init_graphics()

        assets_dir = famnit_gym.envs.DIR_ENVS + '/astronaut/assets/'
        self.unwrapped._sprites['asteroid'] = [
            pygame.image.load(f"{assets_dir}asteroid_1.png").convert_alpha(),
            pygame.image.load(f"{assets_dir}asteroid_2.png").convert_alpha(),
            pygame.image.load(f"{assets_dir}asteroid_3.png").convert_alpha()
        ]
        self._graphics_initialized = True

    def _spawn_asteroid(self):
        rnd = self.unwrapped.np_random
        self._asteroid_back = dict(self._asteroid_front)
        self._asteroid_front['x'] = 1300
        self._asteroid_front['y'] = rnd.integers(200, 600)
        self._asteroid_front['type'] = rnd.integers(0, 3) + 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self._asteroid_front = {
            'x': 0,
            'y': 0,
            'width': 128,
            'height': 128,
            'anchor_x': 64,
            'anchor_y': 64,
            'radius': 64,
            'type': 0
        }

        self._spawn_asteroid()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _simulation_step(self, action: int):
        # Call the base simulation step.
        observation, reward, terminated, truncated, info = self._base_simulation_step(action)

        # determine the time step.
        dt = self.unwrapped._dt

        # Get the astronaut data.
        astronaut = self.unwrapped._astronaut

        # Move asteroids.
        self._asteroid_front['x'] -= self.unwrapped._options['speed'] * dt
        self._asteroid_back['x'] -= self.unwrapped._options['speed'] * dt

        if self._asteroid_front['x'] < astronaut['x']:
            self._spawn_asteroid()

        # Check collision with asteroid.
        dx = self._asteroid_front['x'] - astronaut['x']
        dy = self._asteroid_front['y'] - astronaut['y']
        if math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) < astronaut['radius'] + self._asteroid_front['radius']:
            self.unwrapped._astronaut['collision'] = 'asteroid'
            terminated = True

        # If 'human' rendering mode, paint the frame.
        if self.unwrapped._render_mode == 'human':
            self._paint()

        # Assert the new observation and info.
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def step(self, action: int):
        # If 'human' rendering mode, initialize the graphics.
        if self.unwrapped._render_mode == 'human' and not self._graphics_initialized:
            self._init_graphics()

        observation, reward, terminated, truncated, info = self._simulation_step(action)

        # If 'human' rendering mode, update the frame.
        if self.unwrapped._render_mode == 'human':
            self.unwrapped._update_frame()

        # Compute the reward.
        reward = self.unwrapped._compute_reward(self._get_reward_info())

        return observation, reward, terminated, truncated, info

    def _paint(self):
        surface = self.unwrapped._surface

        # Draw asteroids.
        if self._asteroid_front['type'] > 0:
            surface.blit(
                self.unwrapped._sprites['asteroid'][self._asteroid_front['type'] - 1],
                (
                    round(self._asteroid_front['x'] - self._asteroid_front['anchor_x']),
                    round(800 - self._asteroid_front['y'] - self._asteroid_front['anchor_y']),
                )
            )

        if self._asteroid_back['type'] > 0:
            surface.blit(
                self.unwrapped._sprites['asteroid'][self._asteroid_back['type'] - 1],
                (
                    round(self._asteroid_back['x'] - self._asteroid_back['anchor_x']),
                    round(800 - self._asteroid_back['y'] - self._asteroid_back['anchor_y']),
                )
            )