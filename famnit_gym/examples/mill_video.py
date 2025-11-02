import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove, Video

# The video wrapper will force 'human' render mode anyway.
env = mill.env(render_mode='human')

# We will use the DelayMove wrapper with this example.
env = DelayMove(env, time_limit=10)

# Wrap it also in the Video wrapper to generate a video.
env = Video(env, filename='mill.mp4')

# Reset the environment.
env.reset()

# Execute random actions.
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game is too long!")
        break

    env.step(None)

# When closing, the video will be generated.
env.close()