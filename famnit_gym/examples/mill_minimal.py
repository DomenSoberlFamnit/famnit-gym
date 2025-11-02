import gymnasium as gym
from famnit_gym.envs import mill

# Create and reset the environment.
env = mill.env(render_mode='human')  # 'ansi' for ASCII board.
env.reset()

# Execute random actions.
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    # The game terminates when a player loses.
    if termination:
        print(f"{agent} lost the game!")
        break

    # The game is truncated after 100 moves or when window closed.
    if truncation:
        print("The game is too long!")
        break

    # Legal moves for the current player can be obtained from info.
    # legal_moves = info["legal_moves"]

    # If None is given, a random action is executed.
    env.step(None)

# Close the environment.
env.close()