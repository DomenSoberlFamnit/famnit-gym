from __future__ import annotations
from typing import Any, Optional

import os
import shutil

import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
import pygame

from progress.bar import Bar
import imageio_ffmpeg
import subprocess

class FrameCallback:
    def __init__(self):
        self.frame = 0
        self.frames = []

    def paint(self, surface):
        self.frames.append(surface.copy())
        self.frame += 1

class Video(AECEnv[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], filename='mill.mp4'):
        self._closed = True
        super().__init__()
        
        self.env = env

        if not hasattr(env.unwrapped, "metadata") or "framework" not in env.unwrapped.metadata:
            raise AttributeError(f'The wrapped environment does not belong to the Gymnasium family.')

        if not hasattr(env.unwrapped, "_frame_callback"):
            raise AttributeError(f'The wrapped environment does not allow video rendering.')
        
        self._filename = filename
        self.env.unwrapped._render_mode = 'human'
        self.env.unwrapped._frame_callback = FrameCallback()
        self._ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name != "_cumulative_rewards":
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> AECEnv:
        return self.env.unwrapped

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

    def close(self) -> None:
        if self._closed:
            return

        self.env.close()
        self._closed = True

        env = self.env.unwrapped

        if os.path.exists(self._filename):
            os.remove(self._filename)

        if len(env._frame_callback.frames) > 0:
            fps = env.metadata["render_fps"]
            size = env._frame_callback.frames[0].get_size()
            
            writer = imageio_ffmpeg.write_frames(self._filename, size=size, fps=60, quality=10)
            writer.send(None)
            
            bar = Bar('Generating video', max=len(env._frame_callback.frames))
            for surface in env._frame_callback.frames:
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))
                frame = np.ascontiguousarray(frame)
                writer.send(frame)
                bar.next()
            writer.close()
            bar.finish()
    
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)

    def __str__(self) -> str:
        return f"{type(self).__name__}<{str(self.env)}>"

    def __del__(self):
        if not self._closed:
            self.close()