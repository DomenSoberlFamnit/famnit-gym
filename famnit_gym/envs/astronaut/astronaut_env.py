from typing import Optional
import numpy as np
import gymnasium as gym

import famnit_gym.envs
import famnit_gym.envs.astronaut as astronaut
from famnit_gym.envs.astronaut.reward_function import default_reward_function

class AstronautEnv(gym.Env):
    metadata = {
        "framework": "Gymnasium",
        "name": "famnit_gym/Astronaut-v1",
        "render_modes": ["human"],
        "render_fps": 60
    }

    def get_default_options():
        return {
            'astronaut_x': 300,      # pixels
            'speed': 500,            # pixels / second
            'thrust': 1000,          # pixels / second squared
            'gravity': 400,          # pixels / second squared
            'max_steps': 1200,       # When to truncate the episode.
            'reward_function': None  # A custom reward function.
        }

    def apply_options(user_options):
        options = AstronautEnv.get_default_options()

        for key in user_options:
            options[key] = user_options[key]

        return options

    def __init__(self, render_mode: Optional[str] = None, options: Optional[dict] = None):
        self._render_mode = render_mode

        if options is None:
            self._options = AstronautEnv.get_default_options()
        else:
            self._options = AstronautEnv.apply_options(options)

        self._sprites = {}
        self._graphics_initialized = False

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(2)

        # Constant throughout the whole simulation.
        self._dt = 1.0 / self.metadata['render_fps']

        # Wrappers can set a frame callback, which is called before updating the frame.
        self._frame_callback = None

    def _get_obs(self):
        return np.array(
            [
                self._astronaut['y'],
                self._astronaut['velocity']
            ],
            dtype=np.float32
        )

    def _get_info(self):
        return {
            'variables': ['astronaut_y', 'astronaut_v'],
            'steps': self._steps,
            'collision': self._astronaut['collision']
        }

    def _get_reward_info(self):
        return {
            'collision': False if self._astronaut['collision'] == 'no' else True
        }

    def _init_graphics(self):
        global pygame
        import pygame

        pygame.init()
        self._surface = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Astronaut")
        self._clock = pygame.time.Clock()
        self._graphics_initialized = True

        # Load assets.
        assets_dir = famnit_gym.envs.DIR_ENVS + '/astronaut/assets/'

        self._sprites['landscape'] = {
            'ground': pygame.image.load(f"{assets_dir}ground.png").convert_alpha(),
            'ceiling': pygame.image.load(f"{assets_dir}ceiling.png").convert_alpha()
        }

        self._sprites['astronaut'] = {
            'idle': pygame.image.load(f"{assets_dir}astronaut_idle.png").convert_alpha(),
            'thrust': pygame.image.load(f"{assets_dir}astronaut_thrust.png").convert_alpha()
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        if options is not None:
            self._options = AstronautEnv.apply_options(options)

        self._steps = 0
        
        self._landscape = {
            'x': 0
        }
        
        self._astronaut = {
            'x': self._options['astronaut_x'],
            'y': 400,
            'thrust': False,
            'velocity': 0,
            'acceleration': 0,
            'collision': 'no',
            'width': 70,
            'height': 128,
            'anchor_x': 35,
            'anchor_y': 64,
            'radius': 50
        }

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _simulation_step(self, action: int):
        terminated = False
        truncated = False

        # Determine the action.
        self._astronaut['thrust'] = action != 0

        # determine the time step.
        dt = self._dt

        # Move the landscape.
        self._landscape['x'] -= self._options['speed'] * dt
        if self._landscape['x'] < 0:
            self._landscape['x'] += 1200
        
        # Move the astronaut.
        self._astronaut['acceleration'] = (self._options['thrust'] - self._options['gravity'] if self._astronaut['thrust'] else 0.0 - self._options['gravity'])
        self._astronaut['velocity'] += self._astronaut['acceleration'] * dt
        self._astronaut['y'] += self._astronaut['velocity'] * dt

        # Check collision with ground.
        if self._astronaut['y'] < 50 + self._astronaut['anchor_y']:
            self._astronaut['collision'] = 'ground'
            terminated = True

        # Check collision with ceiling.
        if self._astronaut['y'] > 750 - self._astronaut['anchor_y']:
            self._astronaut['collision'] = 'ceiling'
            terminated = True

        # If 'human' rendering mode, paint the frame.
        if self._render_mode == 'human':
            self._paint()

        # Count the step.
        self._steps += 1

        if self._steps >= self._options['max_steps']:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, 0, terminated, truncated, info

    def step(self, action: int):
        # If 'human' rendering mode, initialize the graphics.
        if self._render_mode == 'human' and not self._graphics_initialized:
            self._init_graphics()

        # Make one simulation step.
        observation, reward, terminated, truncated, info = self._simulation_step(action)

        # If 'human' rendering mode, update the frame.
        if self._render_mode == 'human':
            self._update_frame()

        # Compute the reward.
        reward = self._compute_reward(self._get_reward_info())

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, reward_info):
        reward_function = self._options['reward_function']

        if reward_function is not None:
            return reward_function(reward_info)
        else:
            return default_reward_function(reward_info)

    def _paint(self):
        surface = self._surface

        # Black background
        surface.fill((0, 0, 0))

        # Draw the landscape.
        surface.blit(self._sprites['landscape']['ground'], (round(self._landscape['x']), 700))
        surface.blit(self._sprites['landscape']['ground'], (round(self._landscape['x'] - 1200), 700))
        surface.blit(self._sprites['landscape']['ceiling'], (round(self._landscape['x']), 0))
        surface.blit(self._sprites['landscape']['ceiling'], (round(self._landscape['x'] - 1200), 0))
        
        # Draw the astronaut.
        surface.blit(
            self._sprites['astronaut']['thrust' if self._astronaut['thrust'] else 'idle'],
            (
                round(self._astronaut['x'] - self._astronaut['anchor_x']),
                round(800 - self._astronaut['y'] - self._astronaut['anchor_y']),
            )
        )

    def _update_frame(self):
        if self._frame_callback is not None:
            self._frame_callback.paint(self._surface)

        pygame.display.flip()
        self._clock.tick(self.metadata['render_fps'])

    def close(self):
        if self._graphics_initialized:
            global pygame
            pygame.quit()
