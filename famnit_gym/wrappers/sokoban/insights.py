from typing import Optional

import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np

import famnit_gym.envs.sokoban as sokoban

class Insights(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        super().__init__(env)

        if type(env.unwrapped) is not sokoban.SokobanEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the SokobanEnv class.')
        
        self._env = env.unwrapped
    
    def _get_insights(self):
        # Get the map.
        map = self._env._map.get_array()
        (width, height) = self._env._map.get_map_size()

        # Get tile codes.
        tile_code = self._env._map._tile_code

        # Find map objects.
        crates = []
        goals = []
        finished = []
        (player_x, player_y) = (None, None)

        for y in range(height):
            for x in range(width):
                cell = map[y][x]

                if cell == tile_code['goal']:
                    goals.append([x, y])

                if cell == tile_code['crate'] or cell == tile_code['goal_crate']:
                    crates.append([x, y])
                
                if cell == tile_code['goal_crate']:
                    finished.append([x, y])
                
                if cell == tile_code['player']:
                    player_x = x
                    player_y = y

        # Evaluate possible actions.
        actions_moving = []
        actions_pushing = []
        if (player_x, player_y) is not (None, None):
            for action, (dx, dy) in self._env.action_direction.items():
                x1 = player_x + dx
                y1 = player_y + dy

                # Edge of the map.
                if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                    continue

                # Moving the player to an empty tile.
                if map[y1][x1] == tile_code['floor'] or map[y1][x1] == tile_code['goal']:
                    actions_moving.append(action)
                    continue

                if map[y1][x1] == tile_code['crate'] or map[y1][x1] == tile_code['goal_crate']:
                    # Check if crate can be pushed.
                    x2 = x1 + dx
                    y2 = y1 + dy

                    # Edge of the map.
                    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                        continue
                    
                    # Moving the crate to an empty tile.
                    if map[y2][x2] == tile_code['floor'] or map[y2][x2] == tile_code['goal']:
                        actions_pushing.append(action)

        return {
            'player': np.array([player_x, player_y], dtype=int),
            'crates': np.array(crates, dtype=int),
            'goals': np.array(goals, dtype=int),
            'finished': np.array(finished, dtype=int),
            'actions': {
                'moving': np.array(actions_moving),
                'pushing': np.array(actions_pushing),
            }
        }

    def _augment_info(self, info):
        return info | self._get_insights()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super().reset(seed=seed)
        return observation, self._augment_info(info)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, self._augment_info(info)