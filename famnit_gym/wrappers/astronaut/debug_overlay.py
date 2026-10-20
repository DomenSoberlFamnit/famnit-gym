from typing import Optional
import gymnasium as gym
import famnit_gym
from gymnasium.core import ActType, ObsType
import pygame

import famnit_gym.envs.astronaut as astronaut

class DebugOverlay(gym.Wrapper):
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

        # We keep track of the cumulative reward.
        self._sum_reward = 0

        # Graphics for this wrapper not yet initialized.
        self._graphics_initialized = False

    def _get_obs(self):
        return self._base_get_obs()
        
    def _get_info(self):
        return self._base_get_info()

    def _get_reward_info(self):
        return self._base_get_reward_info()

    def _init_graphics(self):
        global pygame
        import pygame

        self._base_init_graphics()

        # Load fonts.
        assets_dir = famnit_gym.envs.DIR_ENVS + '/astronaut/assets/'
        self._font = pygame.font.Font(f"{assets_dir}JetBrainsMono-Bold.otf", 28)
        self._graphics_initialized = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return super().reset(seed=seed, options=options)
        
        self._sum_reward = 0

    def _simulation_step(self, action: int):
        # Call the base simulation step.
        observation, reward, terminated, truncated, info = self._base_simulation_step(action)
        
        # If 'human' rendering mode, paint the frame.
        if self.unwrapped._render_mode == 'human':
            self._paint()

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
        self._sum_reward += reward

        return observation, reward, terminated, truncated, info

    def _paint(self):
        surface = self.unwrapped._surface
        obs = self._get_obs()
        info = self._get_info()

        # Landscape collision boundaries.
        y = 50 + self.unwrapped._astronaut['anchor_y']
        pygame.draw.line(surface, (255, 0, 0), (0, 50), (1200, 50), 1)
        pygame.draw.line(surface, (255, 0, 0), (0, 750), (1200, 750), 1)

        # Astronaut position
        x = round(self.unwrapped._astronaut['x'])
        y = round(self.unwrapped._astronaut['y'])
        anchor_y = round(self.unwrapped._astronaut['anchor_y'])
        text = self._font.render(f"({x}, {y})", True, (255, 255, 0))
        surface.blit(text, (x - text.get_width() // 2, 800 - y + anchor_y))

        # Astroid position
        if 'asteroid_x' in info['variables']:
            x = round(obs[info['variables'].index('asteroid_x')])
            y = round(obs[info['variables'].index('asteroid_y')])
            text = self._font.render(f"({x}, {y})", True, (255, 255, 0))
            surface.blit(text, (x - round(text.get_width() / 2), 800 - y + 64))

        # Oxygen position
        if 'oxygen_x' in info['variables']:
            x = round(obs[info['variables'].index('oxygen_x')])
            y = round(obs[info['variables'].index('oxygen_y')])
            text = self._font.render(f"({x}, {y})", True, (255, 255, 0))
            surface.blit(text, (x - round(text.get_width() / 2), 800 - y + 40))

        # Cumulative reward
        text = self._font.render(f"Reward: {self._sum_reward:.1f}", True, (255, 255, 0))
        surface.blit(text, (20, 600))

        # Number of steps
        text = self._font.render(f"Steps: {self.unwrapped._steps}", True, (255, 255, 0))
        surface.blit(text, (20, 640))

        # Oxygen count
        if 'oxygen_count' in info:
            text = self._font.render(f"Oxygen: {info['oxygen_count']}", True, (255, 255, 0))
            surface.blit(text, (20, 560))