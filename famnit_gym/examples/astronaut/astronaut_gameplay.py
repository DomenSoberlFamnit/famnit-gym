import gymnasium as gym
import famnit_gym

from famnit_gym.wrappers.astronaut import Asteroids
from famnit_gym.wrappers.astronaut import Oxygen
from famnit_gym.wrappers.astronaut import DebugOverlay
from famnit_gym.wrappers.astronaut import Video
from famnit_gym.wrappers.astronaut import Keyboard

# Create and reset the environment.
env = gym.make('famnit_gym/Astronaut-v1', render_mode='human')

# Stack the additional (optional) functionalities.
env = Asteroids(env)     # Add the asteroids to the game.
env = Oxygen(env)        # Add the oxygen tanks to the game.
env = DebugOverlay(env)  # Additional information on screen.
env = Video(env, filename='astronaut.webm')  # Record a video.

# Let the keyboard be the top wrapper, to assert keyboard
# actions to the underlying environments. Then use:
#   SPACE - thrust, ENTER - reset, ESC - exit, P - pause
env = Keyboard(env)

# Reset the environment.
observation, info = env.reset()

# Play the game.
while True:
    observation, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break

# Close the environment.
env.close() 