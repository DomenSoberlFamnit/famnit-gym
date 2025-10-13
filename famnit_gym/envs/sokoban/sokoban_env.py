from typing import Optional
import numpy as np
import gymnasium as gym

import famnit_gym.envs
import famnit_gym.envs.sokoban as sokoban

class SokobanEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    action_direction = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0)
    }

    def __init__(self, render_mode: Optional[str] = None, options: Optional[dict] = None):
        self._render_mode = render_mode
        map_template = None
        scale = None

        if options is not None:
            if 'map_template' in options:
                map_template = options['map_template']
            if 'scale' in options:
                scale = options['scale']
        
        self._pygame_initialized = False

        self._map = sokoban.SokobanMap(
            map_template=map_template,
            scale=scale,
            dir=famnit_gym.envs.DIR_ENVS + '/sokoban'
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=5,
            shape=self._map.get_map_size(),
            dtype=np.uint8
        )
        
        self.action_space = gym.spaces.Discrete(4)
    
    def _get_obs(self):
        return self._map.get_array()

    def _get_info(self):
        return {
            'steps': self._steps
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._map.reset()
        self._steps = 0

        # If 'human' rendering mode, initialize pygame.
        if self._render_mode == 'human':
            if not self._pygame_initialized:
                global pygame
                import pygame
            
                pygame.init()
                self._surface = pygame.display.set_mode(self._map.window_size())
                self._clock = pygame.time.Clock()
                self._pygame_initialized = True
            
            self._map.paint(self._surface)
            pygame.display.flip()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        terminated = False
        truncated = False
        
        reward = 0

        if action < 0 or action > 3:
            observation = self._get_obs()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

        animate = self._render_mode == 'human'

        (dx, dy) = self.action_direction[action]
        self._map.move_player(dx, dy, animate=animate)

        if animate:
            global pygame
            while self._map.animation_running() and not truncated:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        truncated = True

                self._map.animate_step()
                self._map.paint(self._surface)
                pygame.display.flip()
                self._clock.tick(self.metadata['render_fps'])
        
        self._steps += 1
        
        terminated = self._map.game_finished()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
