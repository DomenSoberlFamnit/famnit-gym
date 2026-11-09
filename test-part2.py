import time
import random
from famnit_gym.envs import mill
from mcts import mcts_search

# 1. Define the 3 Difficulties
DIFFICULTIES = {
    "Easy": {
        "iterations": 25,
        "depth": 10,
        "blunder_rate": 0.20  # 20% chance of making a random move instead of MCTS
    },
    "Medium": {
        "iterations": 150,
        "depth": 30,
        "blunder_rate": 0.0
    },
    "Hard": {
        "iterations": 400,
        "depth": 50,
        "blunder_rate": 0.0
    }
}

# Trackers for tracking the computation time it took for each difficulty level across all games played
time_stats = {
    "Easy":   {"total_time": 0.0, "total_moves": 0},
    "Medium": {"total_time": 0.0, "total_moves": 0},
    "Hard":   {"total_time": 0.0, "total_moves": 0},
}


def choose_move(model, player, difficulty_name):
    """Selects a move based on difficulty settings and records the time taken."""
    cfg = DIFFICULTIES[difficulty_name]
    legal_moves = model.legal_moves(player=player)

    if not legal_moves:
        return None

    # Measure the computation time
    start_time = time.perf_counter()

    # Check for blunder (Easy mode AI has a chance to make a random move)
    if random.random() < cfg["blunder_rate"]:
        move = random.choice(legal_moves)
    else:
        move = mcts_search(
            root_model=model,
            player=player,
            iterations=cfg["iterations"],
            max_simulation_depth=cfg["depth"]
        )

    duration = time.perf_counter() - start_time

    time_stats[difficulty_name]["total_time"] += duration
    time_stats[difficulty_name]["total_moves"] += 1

    return move


def play_game(diff_p1, diff_p2):
    """Plays a single game between two difficulties and returns the winner (1, 2, or 0 for draw)."""
    env = mill.env()
    env.reset()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if env.rewards["player_1"] > 0:
                winner = 1
            elif env.rewards["player_2"] > 0:
                winner = 2
            else:
                winner = 0
            env.close()
            return winner

        model = mill.transition_model(env)
        player = 1 if agent == "player_1" else 2
        difficulty = diff_p1 if player == 1 else diff_p2

        move = choose_move(model, player, difficulty)
        env.step(move)


def run_tournament(games_per_matchup=6):
    """Runs matches between Easy vs Medium, Medium vs Hard, and Easy vs Hard."""
    matchups = [
        ("Easy", "Medium"),
        ("Medium", "Hard"),
        ("Easy", "Hard")
    ]

    print("=" * 60)
    print("Starting Tournament: Easy vs Medium vs Hard")
    print("=" * 60)

    for diff_a, diff_b in matchups:
        print(f"\n--- Matchup: {diff_a} vs. {diff_b} ({games_per_matchup} games) ---")
        score = {diff_a: 0, diff_b: 0, "Draw": 0}

        for game_idx in range(games_per_matchup):
            # Alternate who plays as Player 1 to make it fair
            if game_idx % 2 == 0:
                # diff_a is Player 1, diff_b is Player 2
                winner = play_game(diff_p1=diff_a, diff_p2=diff_b)
                if winner == 1:
                    score[diff_a] += 1
                    winner_name = diff_a
                elif winner == 2:
                    score[diff_b] += 1
                    winner_name = diff_b
                else:
                    score["Draw"] += 1
                    winner_name = "Draw"
                print(f"  Game {game_idx+1}: [P1: {diff_a}] vs [P2: {diff_b}] --> Winner: {winner_name}")
            else:
                # diff_b is Player 1, diff_a is Player 2
                winner = play_game(diff_p1=diff_b, diff_p2=diff_a)
                if winner == 1:
                    score[diff_b] += 1
                    winner_name = diff_b
                elif winner == 2:
                    score[diff_a] += 1
                    winner_name = diff_a
                else:
                    score["Draw"] += 1
                    winner_name = "Draw"
                print(f"  Game {game_idx+1}: [P1: {diff_b}] vs [P2: {diff_a}] --> Winner: {winner_name}")

        print(f"Result: {diff_a}: {score[diff_a]} wins | {diff_b}: {score[diff_b]} wins | Draws: {score['Draw']}")

    # Print the timing exploration results
    print("\n" + "=" * 60)
    print("Computation Time Statistics Across All Games:")
    print("=" * 60)
    print(f"{'Difficulty':<10} | {'Total Moves':<12} | {'Avg Time / Move':<15}")
    print("-" * 45)
    for diff in ["Easy", "Medium", "Hard"]:
        stats = time_stats[diff]
        moves = stats["total_moves"]
        avg_time = (stats["total_time"] / moves) if moves > 0 else 0.0
        print(f"{diff:<10} | {moves:<12} | {avg_time:.4f} seconds")


if __name__ == "__main__":
    run_tournament(games_per_matchup=6)