import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.sokoban import Keyboard

# Create and reset the environment.
env = gym.make('famnit_gym/Sokoban-v1', render_mode='human')

# Wrap the environment into the Keyboard wrapper.
env = Keyboard(env)

# Reset the environment.
observation, info = env.reset()

# Run until game over,
done = False
while not done:
    # Use input will be used instead of the given action.
    _, _, terminated, truncated, _ = env.step(0)

    # The episode is truncated if the user quits the game.
    done = terminated or truncated