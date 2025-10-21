import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import DelayMove

env = mill.env(render_mode='human')

# Wrap the Mill environment into the DelayMove wrapper
# to have a pause between the moves.
# Set the time limit or omit it for 5 seconds default.
env = DelayMove(env, time_limit=10)

env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game is too long!")
        break

    # The step method will wait for the key or mouse press,
    # but not more than the set time limit, after which it
    # will execute the given move.
    env.step(None)