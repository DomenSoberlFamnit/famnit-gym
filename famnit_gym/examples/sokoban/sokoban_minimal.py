import gymnasium as gym
import famnit_gym

# Create and reset the environment.
env = gym.make('famnit_gym/Sokoban-v1', render_mode='human')
observation, info = env.reset()

# Execute random actions.
done = False
while not done:
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Close the environment.
env.close()