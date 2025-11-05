from __future__ import annotations

from typing import Any

import pygame
import gymnasium.spaces
import numpy as np

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from famnit_gym.envs.mill import MillEnv

class UserInteraction(AECEnv[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        super().__init__()

        if type(env.unwrapped) is not MillEnv:
            raise AttributeError(f'The wrapped environment must be an instance of the MillEnv class.')

        self.env = env

        # Get the FPS, the surface and the clock.
        self._fps = self.env.unwrapped.metadata['render_fps']
        self._surface = self.env.unwrapped._surface
        self._clock = self.env.unwrapped._clock
        
        # The position over which the mouse currently resides.
        self._selection_color = (128, 128, 255, 192)

        # Marked positions.
        self._markers = [None for _ in range(25)]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name != "_cumulative_rewards":
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> AECEnv:
        return self.env.unwrapped

    def set_selection_color(self, color: tuple) -> None:
        self._selection_color = color

    def mark_position(self, position: int, color: tuple) -> None:
        self._markers[position] = color

    def unmark_position(self, position: int) -> None:
        self._markers[position] = None
    
    def clear_markings(self) -> None:
        self._markers = [None for _ in range(25)]

    def _selection_from_mouse(self, pos) -> (int, int, int):
        (x, y) = pos
        selected = 0
        sel_x, sel_y = 0, 0
        
        # Iterate through all positions and find over which one the mouse hovers.
        for (i, (row, col)) in enumerate(self.env.unwrapped._render_positions):
            (sel_x, sel_y) = (52 + col * 100, 52 + row * 100)  # Central point
            if x >= sel_x - 30 and x <= sel_x + 30 and y >= sel_y - 30 and y <= sel_y + 30:
                selected = i + 1
                break
        
        return selected, sel_x, sel_y

    def interact(self) -> dict:
        # This wrapper only has effect in human render mode.
        if not self.env.unwrapped.render_mode == 'human':
            return {}
        
        # Unwrap the environment.
        mill = self.env.unwrapped

        # Get the selection from the current mouse position.
        selected, sel_x, sel_y = self._selection_from_mouse(pygame.mouse.get_pos())

        # Interract with the board until a recognized event is triggered.
        message = None
        while message is None:
            for event in pygame.event.get():
                # If the user quits.
                if event.type == pygame.QUIT:
                    message = {'type': 'quit'}
                
                # If a mouse has been moved, check which position is selected.
                elif event.type == pygame.MOUSEMOTION:
                    new_selected, new_sel_x, new_sel_y = self._selection_from_mouse(event.pos)
                    
                    # The selection has changed.
                    if new_selected != selected:
                        selected = new_selected
                        sel_x, sel_y = new_sel_x, new_sel_y
                        message = {
                            'type': 'mouse_move',
                            'position': selected
                        }
                    
                # If a mouse button has been clicked.
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    selected, _, _ = self._selection_from_mouse(event.pos)
                    message = {
                        'type': 'mouse_click',
                        'position': selected
                    }
                
                # If a key kas been pressed
                elif event.type == pygame.KEYDOWN:
                    message = {
                        'type': 'key_press',
                        'key': pygame.key.name(event.key)
                    }

            # Draw the board.
            mill._paint_board()

            # Mark the marked positions.
            for (position, color) in enumerate(self._markers):
                if color is not None:
                    (row, col) = mill._render_positions[position - 1]
                    (x, y) = (52 + col * 100, 52 + row * 100)
                    pygame.gfxdraw.filled_circle(self._surface, x, y, 35, color)
                    pygame.gfxdraw.aacircle(self._surface, x, y, 35, color)

            # Mark the selected position.
            if selected != 0:
                pygame.gfxdraw.filled_circle(self._surface, sel_x, sel_y, 35, self._selection_color)
                pygame.gfxdraw.aacircle(self._surface, sel_x, sel_y, 35, self._selection_color)

            # Draw the pieces and finish drawing.
            mill._paint_pieces()

            # Update the frame.
            self.env.unwrapped._update_frame()

            # Wait one frame.
            self._clock.tick(self._fps)

        # Return the message
        return message

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
        self.env.step(action)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def __str__(self) -> str:
        return f"{type(self).__name__}<{str(self.env)}>"