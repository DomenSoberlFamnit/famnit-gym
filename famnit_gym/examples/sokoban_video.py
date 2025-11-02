import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.sokoban import Keyboard, Video

# Create and reset the environment.
env = gym.make('famnit_gym/Sokoban-v1', render_mode='human')

# We will use the Keyboard wrapper with this example.
env = Keyboard(env)

# Wrap it also in the Video wrapper to generate a video.
env = Video(env, filename='sokoban.mp4')

# Reset the environment.
observation, info = env.reset()

# Run until game over.
done = False
while not done:
    _, _, terminated, truncated, _ = env.step(0)

    done = terminated or truncated

# When closing, the video will be generated.
env.close()