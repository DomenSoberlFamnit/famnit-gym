from typing import Optional
import numpy as np
import gymnasium as gym
import famnit_gym
from gymnasium.core import ActType, ObsType
import pygame

import famnit_gym.envs.astronaut as astronaut

class Oxygen(gym.Wrapper):
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
                self._oxygen_front['x'],
                self._oxygen_front['y']
            ],
            dtype=np.float32
        )
        
        return np.concatenate([self._base_get_obs(), obs])
        
    def _get_info(self):
        info = self._base_get_info()
        info['oxygen_picked'] = 'yes' if self._oxygen_picked else 'no'
        info['oxygen_count'] = self._oxygen_count
        info['variables'] = info['variables'] + ['oxygen_x', 'oxygen_y']
        
        return info

    def _get_reward_info(self):
        info = self._get_info()
        reward_info = self._base_get_reward_info()
        reward_info['oxygen_picked'] = True if info['oxygen_picked'] == 'yes' else False

        return reward_info

    def _init_graphics(self):
        global pygame
        import pygame

        self._base_init_graphics()

        assets_dir = famnit_gym.envs.DIR_ENVS + '/astronaut/assets/'
        self.unwrapped._sprites['oxygen'] = pygame.image.load(f"{assets_dir}oxygen.png").convert_alpha()
        self._graphics_initialized = True

    def _spawn_oxygen(self, offset_x=0):
        rnd = self.unwrapped.np_random
        self._oxygen_back = dict(self._oxygen_front)
        self._oxygen_front['x'] = 1300 + offset_x
        self._oxygen_front['y'] = rnd.integers(200, 600)
        self._oxygen_front['seen'] = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self._oxygen_front = {
            'x': 0,
            'y': 0,
            'width': 36,
            'height': 92,
            'anchor_x': 18,
            'anchor_y': 46,
            'seen': False
        }

        self._spawn_oxygen(offset_x=500)
        
        self._oxygen_picked = False
        self._oxygen_count = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _simulation_step(self, action: int):
        # Call the base simulation step.
        observation, reward, terminated, truncated, info = self._base_simulation_step(action)

        # We don't know yet if the oxygen tank has been picked.
        self._oxygen_picked = False

        # determine the time step.
        dt = self.unwrapped._dt

        # Get the astronaut data.
        astronaut = self.unwrapped._astronaut

        # Move oxygen.
        self._oxygen_front['x'] -= self.unwrapped._options['speed'] * dt
        self._oxygen_back['x'] -= self.unwrapped._options['speed'] * dt

        if self._oxygen_front['x'] < self.unwrapped._astronaut['x']:
            self._spawn_oxygen()

        # Check collision with oxygen.
        if self._oxygen_front['seen']:
            if self._oxygen_front['x'] < astronaut['x'] + astronaut['anchor_x']:
                if self._oxygen_front['y'] - self._oxygen_front['anchor_y'] < astronaut['y'] + astronaut['anchor_y']:
                    if self._oxygen_front['y'] + self._oxygen_front['anchor_y'] > astronaut['y'] - astronaut['anchor_y']:
                        self._oxygen_picked = True
                        self._oxygen_count += 1
                        self._oxygen_front['seen'] = False

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

        # Draw oxygen.
        if self._oxygen_front['seen']:
            surface.blit(
                self.unwrapped._sprites['oxygen'],
                (
                    round(self._oxygen_front['x'] - self._oxygen_front['anchor_x']),
                    round(800 - self._oxygen_front['y'] - self._oxygen_front['anchor_y']),
                )
            )

        if self._oxygen_back['seen']:
            surface.blit(
                self.unwrapped._sprites['oxygen'],
                (
                    round(self._oxygen_back['x'] - self._oxygen_back['anchor_x']),
                    round(800 - self._oxygen_back['y'] - self._oxygen_back['anchor_y']),
                )
            )