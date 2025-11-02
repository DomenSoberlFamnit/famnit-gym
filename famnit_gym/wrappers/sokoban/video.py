import os
import shutil
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType
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

class Video(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType], filename='sokoban.mp4'):
        self._closed = True
        super().__init__(env)
        
        if not hasattr(env.unwrapped, "metadata") or "framework" not in env.unwrapped.metadata:
            raise AttributeError(f'The wrapped environment does not belong to the Gymnasium family.')

        if not hasattr(env.unwrapped, "_frame_callback"):
            raise AttributeError(f'The wrapped environment does not allow video rendering.')
        
        self._env = env.unwrapped
        self._filename = filename
        self._env._render_mode = 'human'
        self._env._frame_callback = FrameCallback()
        self._ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
        self._closed = False
        
    def close(self):
        if self._closed:
            return

        self.env.close()
        self._closed = True

        if os.path.exists(self._filename):
            os.remove(self._filename)

        if len(self._env._frame_callback.frames) > 0:
            fps = self._env.metadata["render_fps"]
            size = self._env._frame_callback.frames[0].get_size()
            
            writer = imageio_ffmpeg.write_frames(self._filename, size=size, fps=60, quality=10)
            writer.send(None)
            
            bar = Bar('Generating video', max=len(self._env._frame_callback.frames))
            for surface in self._env._frame_callback.frames:
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))
                frame = np.ascontiguousarray(frame)
                writer.send(frame)
                bar.next()
            writer.close()
            bar.finish()
    
    def __del__(self):
        if not self._closed:
            self.close()