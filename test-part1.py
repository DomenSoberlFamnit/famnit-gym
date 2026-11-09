import random
from famnit_gym.envs import mill
from mcts import mcts_search

def play_match(agent_1_type, agent_2_type, c1=1.414, c2=1.414, iters=200):
    env = mill.env()
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            # PettingZoo reward: +1 win, -1 loss, 0 draw
            winner = 1 if env.rewards["player_1"] > 0 else (2 if env.rewards["player_2"] > 0 else 0)
            env.close()
            return winner

        model = mill.transition_model(env)
        player = 1 if agent == "player_1" else 2
        agent_type = agent_1_type if player == 1 else agent_2_type
        c_val = c1 if player == 1 else c2

        if agent_type == "random":
            moves = model.legal_moves(player)
            move = random.choice(moves)
        else: # "mcts"
            move = mcts_search(model, player=player, iterations=iters, c_param=c_val)

        env.step(move)

# 1. Test MCTS vs Random Moves Player (10 games)
print("=== Running MCTS (Player 1) vs Random (Player 2) ===")
wins = {1: 0, 2: 0, 0: 0}
for game in range(10):
    w = play_match("mcts", "random", iters=200)
    wins[w] += 1
    print(f"Game {game+1}: Winner = Player {w}")
print(f"Summary: MCTS Wins: {wins[1]}, Random Wins: {wins[2]}, Draws: {wins[0]}\n")

# 2. Test Different UCT c-values: c=0.2 (Greedy) vs c=1.414 (Balanced)
print("=== Running UCT Comparison: c=0.2 (Player 1) vs c=1.414 (Player 2) ===")
c_wins = {1: 0, 2: 0, 0: 0}
for game in range(6):
    w = play_match("mcts", "mcts", c1=0.2, c2=1.414, iters=150)
    c_wins[w] += 1
    print(f"Game {game+1}: Winner = Player {w}")
print(f"Summary: c=0.2 Wins: {c_wins[1]}, c=1.414 Wins: {c_wins[2]}, Draws: {c_wins[0]}")