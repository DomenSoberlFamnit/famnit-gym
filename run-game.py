import gymnasium as gym
from famnit_gym.envs import mill
from mcts import mcts_search

env = mill.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break

    initial_model = mill.transition_model(env)
    player = 1 if agent == "player_1" else 2

    move = mcts_search(initial_model, player, iterations=300)

    env.step(move)

env.close()