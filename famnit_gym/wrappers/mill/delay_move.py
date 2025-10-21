from __future__ import annotations

from typing import Any

import pygame
import gymnasium.spaces
import numpy as np

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from famnit_gym.envs.mill import MillEnv

class DelayMove(AECEnv[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], time_limit=None):
        super().__init__()

        if type(env.unwrapped) is not MillEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the MillEnv class.')

        self.env = env

        # Get the FPS and the clock.
        self._fps = self.env.unwrapped.metadata['render_fps']
        self._clock = self.env.unwrapped._clock

        # Set the time limit. Default is 5 seconds.
        self._time_limit = time_limit if time_limit is not None else 5

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name != "_cumulative_rewards":
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> AECEnv:
        return self.env.unwrapped

    def close(self) -> None:
        self.env.close()

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)

    def state(self) -> np.ndarray:
        return self.env.state()

    def step(self, action: ActionType) -> None:
        # This wrapper only has effect in human render mode.
        if self.env.unwrapped.render_mode == 'human':
            time = 0
            wait = True
            
            # Wait until a key pressed, mouse clicked, or time is up.
            while wait:
                for event in pygame.event.get():
                    # If the user quits, just truncate and return.
                    if event.type == pygame.QUIT:
                        self.env.unwrapped.truncations = {agent: True for agent in self.agents}
                        return
                    
                    # Check the keyboard.
                    if event.type == pygame.KEYDOWN:
                        wait = False
                    
                    # Check mouse click.
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        wait = False
                    
                # Check the timer.
                if time >= self._time_limit:
                    wait = False
                
                # Measure time.
                time += 1 / self._fps

                # Wait one frame.
                self._clock.tick(self._fps)
            
        # Execute the action.
        self.env.step(action)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def __str__(self) -> str:
        return f"{type(self).__name__}<{str(self.env)}>"